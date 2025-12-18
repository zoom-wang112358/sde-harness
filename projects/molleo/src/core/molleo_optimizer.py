"""Main MolLEO optimizer using SDE harness framework"""

import random
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rdkit import Chem
import sys
import os
from concurrent.futures import ThreadPoolExecutor, wait
# Add SDE harness to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Generation, Workflow
from ..utils import (
    mol_to_smiles, 
    smiles_to_mol,
    make_mating_pool, 
    reproduce,
    get_best_mol
)
from ..oracles import MolecularOracle
from .prompts import MolecularPrompts
from openai import OpenAI

def topk_auc_from_history(history, top_k, finish, freq_log, max_oracle_calls):

    n = len(history)
    if n == 0 or max_oracle_calls == 0:
        return 0.0

    ordered = sorted(history, key=lambda x: x["call_count"])

    area = 0.0
    prev = 0.0
    called = 0

    upper = min(n, max_oracle_calls)
    for idx in range(freq_log, upper, freq_log):
        prefix = ordered[:idx]
        top_now = sorted(prefix, key=lambda x: x["score"], reverse=True)[:top_k]
        top_k_now = np.mean([item["score"] for item in top_now])

        area += freq_log * (top_k_now + prev) / 2
        prev = top_k_now
        called = idx

    top_all = sorted(ordered, key=lambda x: x["score"], reverse=True)[:top_k]
    top_k_now = np.mean([item["score"] for item in top_all])

    area += (n - called) * (top_k_now + prev) / 2

    if finish and n < max_oracle_calls:
        area += (max_oracle_calls - n) * top_k_now

    return area / max_oracle_calls

class MolLEOOptimizer:
    """
    MolLEO: LLM-augmented evolutionary algorithm for molecular discovery
    Integrated with SDE harness framework
    """
    
    def __init__(self, 
                 oracle: MolecularOracle,
                 oracle_name: str,
                 population_size: int = 100,
                 offspring_size: int = 200,
                 mutation_rate: float = 0.01,
                 n_jobs: int = -1,
                 model_name: str = "openai/gpt-4o-2024-08-06",
                 freq_log: int = 100,
                 max_oracle_calls: int = 10000,
                 patience: int = 5,
                 seed: int = 42,
                 output_dir: Optional[str] = None,
                 use_llm_mutations: bool = True):
        """
        Initialize MolLEO optimizer
        
        Args:
            oracle: Molecular property oracle
            population_size: Size of population to maintain
            offspring_size: Number of offspring per generation
            mutation_rate: Probability of mutation
            n_jobs: Number of parallel jobs
            model_name: LLM model to use for mutations
            use_llm_mutations: Whether to use LLM for guided mutations
        """
        self.oracle = oracle
        self.oracle_name = oracle_name
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.n_jobs = n_jobs
        self.model_name = model_name
        self.freq_log = freq_log
        self.max_oracle_calls = max_oracle_calls
        self.patience = patience
        self.seed = seed
        self.output_dir = output_dir
        random.seed(seed)
        np.random.seed(seed)
        self.use_llm_mutations = use_llm_mutations
        
        # Initialize generation model
        self.generator = Generation(
            models_file=os.path.join(project_root, "models.yaml"),
            credentials_file=os.path.join(project_root, "credentials.yaml")
        )
        
        # Initialize population
        self.population_mol = []
        self.population_scores = []
        self.generation_count = 0
        self.all_results = {}
        
    def initialize_population(self, starting_smiles: List[str]):
        """Initialize population from starting molecules"""
        self.population_mol = []
        self.population_scores = []
        
        for smiles in starting_smiles:
            mol = smiles_to_mol(smiles)
            if mol is not None:
                score = self.oracle.evaluate_molecule(smiles)
                self.population_mol.append(mol)
                self.population_scores.append(score)
                self.all_results[smiles] = score
                
        # Fill remaining population with random mutations
        while len(self.population_mol) < self.population_size:
            parent_idx = random.randint(0, len(self.population_mol) - 1)
            parent_mol = self.population_mol[parent_idx]
            
            # Try mutation
            if self.use_llm_mutations:
                mutant = self._llm_mutate(parent_mol)
            else:
                mutant = self._random_mutate(parent_mol)
                
            if mutant is not None:
                mutant_smiles = mol_to_smiles(mutant)
                if mutant_smiles not in self.all_results:
                    score = self.oracle.evaluate_molecule(mutant_smiles)
                    self.population_mol.append(mutant)
                    self.population_scores.append(score)
                    self.all_results[mutant_smiles] = score
                    
    def _llm_mutate(self, parent_mols, parent_scores, mutation_rate) -> Optional[Chem.Mol]:
        """Use LLM to generate molecular mutations with context"""
        parent_smiles = [mol_to_smiles(mol) for mol in parent_mols]
        
        # If population is not yet established, use simple mutation prompt
        if not self.population_scores:
            prompt = MolecularPrompts.get_mutation_prompt(
                parent_smiles=parent_smiles,
                num_mutations=5
            )
        else:
            # Get top molecules from current population for context
            #top_indices = np.argsort(self.population_scores)[-5:][::-1]
            context_molecules = []
            for idx in range(len(parent_mols)):
                mol = parent_mols[idx]
                score = parent_scores[idx]
                smiles = mol_to_smiles(mol)
                context_molecules.append(f"[{smiles}, {score:.4f}]")
            
            # Create context-aware prompt
            molecule_context = "\n".join(context_molecules)
            
            # Use optimization prompt for better context
            prompt = MolecularPrompts.get_optimization_prompt(
                molecule_data=molecule_context,
                target_property=self.oracle.property_name,
            )
            prompt=prompt.build()
        for i in range(3):  # Try up to 3 times
            try:
                system_message = "You are a chemical specialist who can analyze and optimize molecules."
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                # Generate mutations using SDE harness Generation
                if self.model_name == 'openai/gpt-5':
                    
                    response = self.generator.generate(
                        messages=messages,
                        model_name=self.model_name,
                        temperature=1,
                        reasoning_effort="high",
                        max_tokens=32000,
                    )
                    

                elif self.model_name == 'openai/gpt-5-chat-latest':
                    response = self.generator.generate(
                        messages=messages,
                        model_name=self.model_name,
                        temperature=1,
                        max_tokens=16384,
                    )
                else:
                    response = self.generator.generate(
                        messages=messages,
                        model_name=self.model_name,
                        temperature=0.8,
                        max_tokens=8192,
                    )
                #print(response)
                #print(f"LLM Mutation Response: {response['text'].strip()}")
                # Parse response - handle various formats
                if isinstance(response, str):
                    text = response.strip()
                else:
                    text = response['text'].strip()
                
                # Try to extract SMILES from various formats
                crossover_smiles = []
                
                # Check for boxed format (like original GPT4 implementation)
                proposed_smiles = re.search(r'<box>(.*?)</box>', text, re.DOTALL).group(1)
                proposed_smiles = sanitize_smiles(proposed_smiles)
                assert proposed_smiles is not None
                print('LLM proposed SMILES:', proposed_smiles)
                mol = Chem.MolFromSmiles(proposed_smiles, sanitize=True)
                if mol is not None:
                    return mol
                else:
                    print(f"Invalid SMILES from LLM: {proposed_smiles}")
                    continue    
                            
            except Exception as e:
                print(f"LLM mutation failed: {e}")
                continue
        
        try:
            new_child = self._random_crossover(parent_mols[0], parent_mols[1])
            if new_child is not None:
                new_child = self._random_mutate(new_child)
            print(f"GA New child SMILES: {Chem.MolToSmiles(new_child)}")
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = parent_mol[0]    
        # Fallback to random mutation
        return new_child

    def _random_crossover(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[Chem.Mol]:
        """Random molecular mutation using original MolLEO operations"""
        from src.ga import crossover as co
        # Use original random mutation
        return co.crossover(mol1, mol2)  
    def _random_mutate(self, parent_mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Random molecular mutation using original MolLEO operations"""
        from src.ga import mutations as mu
        # Use original random mutation
        return mu.mutate(parent_mol, self.mutation_rate, mol_lm=None)
        
    def evolve_one_generation(self):
        """Evolve population for one generation"""
        self.generation_count += 1
        
        # Create mating pool
        mating_pool = make_mating_pool(
            self.population_mol,
            self.population_scores,
            self.offspring_size
        )
        
        # Generate offspring
        offspring_mol = []
        offspring_scores = []
        
        successful_reproductions = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(reproduce, mating_pool, self.mutation_rate, mol_lm=self) for _ in range(self.offspring_size)]
            generated_mol = [future.result() for future in futures]

        for mol in generated_mol:
            if mol is not None:
                successful_reproductions += 1
                child_smiles = mol_to_smiles(mol)
                
                # Check if already evaluated
                if child_smiles in self.all_results:
                    score = self.all_results[child_smiles]
                    continue
                else:
                    if self.oracle.in_history(child_smiles):
                        continue
                    # Evaluate new molecule
                    score = self.oracle.evaluate_molecule(child_smiles)
                    self.all_results[child_smiles] = score
                offspring_mol.append(mol)
                offspring_scores.append(score)
            else:
                print(f"DEBUG: Child is None")
        
        print(f"DEBUG: Successful reproductions: {successful_reproductions}")
        print(f"DEBUG: Offspring generated: {len(offspring_mol)}")
        
        # Combine population and offspring
        all_mol = self.population_mol + offspring_mol
        all_scores = self.population_scores + offspring_scores
        
        # Select top molecules for next generation
        sorted_indices = np.argsort(all_scores)[::-1][:self.population_size]
        self.population_mol = [all_mol[i] for i in sorted_indices]
        self.population_scores = [all_scores[i] for i in sorted_indices]
        
    def optimize(self, 
                 starting_smiles: List[str],
                 num_generations: int = 20) -> Dict[str, Any]:
        """
        Run optimization
        
        Args:
            starting_smiles: Initial molecules
            num_generations: Number of generations to run
            
        Returns:
            Optimization results
        """
        # Initialize population
        self.initialize_population(starting_smiles)
        
        # Track best molecules
        best_scores = []
        best_molecules = []
        patience = 0
        no_improve_gen = 0
        if len(self.population_scores) > 100:
            old_score = np.mean(self.population_scores)
        else:
            old_score = 0
        # Evolution loop
        for gen in range(num_generations):
            # Evolve
            self.evolve_one_generation()
            
            # Track best
            best_idx = np.argmax(self.population_scores)
            best_score = self.population_scores[best_idx]
            best_mol = self.population_mol[best_idx]
            best_smiles = mol_to_smiles(best_mol)
            
            best_scores.append(best_score)
            best_molecules.append(best_smiles)
            top_k_auc = topk_auc_from_history(self.oracle.history, top_k=10, finish=True, freq_log=self.freq_log, max_oracle_calls=self.max_oracle_calls)
            print(f"Generation {gen+1}: Best score = {best_score:.4f}, Top-10 AUC = {top_k_auc:.4f}, "
                f"Oracle calls = {self.oracle.call_count}")
            # Early stopping based on oracle calls
            if self.oracle.call_count >= self.max_oracle_calls:
                print(f"Reached max oracle calls: {self.oracle.call_count}. Stopping optimization.")
                break
            # Early stopping based on patience
            if len(self.oracle.history) >= 100:
                new_score = np.mean(self.population_scores)
                if (new_score - old_score) < 1e-3:
                    no_improve_gen += 1
                    if no_improve_gen >= self.patience:
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    no_improve_gen = 0
                old_score = new_score
            
        # Get final results
        final_population = [
            (mol_to_smiles(mol), score) 
            for mol, score in zip(self.population_mol, self.population_scores)
        ]
        import json
        metrics_dir = os.path.join(self.output_dir, "optimizition_results")
        os.makedirs(metrics_dir, exist_ok=True) 
        output_file = os.path.join(
            metrics_dir,
            f"results_{self.oracle_name}_{self.model_name.replace('/', '_') if self.model_name else 'random'}_{self.seed}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(self.oracle.history, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        return {
            "best_molecule": best_molecules[-1],
            "best_score": best_scores[-1],
            "best_scores_history": best_scores,
            "best_molecules_history": best_molecules,
            "final_population": final_population,
            "top_k_auc": top_k_auc,
            "oracle_calls": self.oracle.call_count,
            "all_results": self.all_results,
        }
    
    # Compatibility method for mutation
    def mutate(self, parent_mols, parent_scores, mutation_rate) -> Optional[Chem.Mol]:
        """Mutate molecule (for compatibility with original code)"""
        if self.use_llm_mutations:
            return self._llm_mutate(parent_mols, parent_scores, mutation_rate)
        else:
            return self._random_mutate(parent_mols[0])

def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, canonical=True)
        return smi_canon
    except:
        return None