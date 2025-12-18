"""Prompts for molecular generation and optimization"""

import sys
import os

# Add SDE harness to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Prompt

Property_description = {
    "jnk3": "The JNK3 score measures the inhibitory ability of a molecule against c-Jun N-terminal kinase 3 (JNK3).",
    "gsk3b": "The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.",
    "drd2": "The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).",
    "celecoxib_rediscovery": "The celecoxib rediscovery score measures the tanimoto similarity of a molecule to Celecoxib, a known drug.",
    "sitagliptin_mpo": "The sitagliptin MPO score is calculated as the geometric mean of four component metrics: \
    1. Tanimoto similarity to Sitagliptin, a known drug. \
    2. Have similar Molecular logP to sitagliptin. \
    3. Have similar Topological polar surface area (TPSA) to sitagliptin. \
    4. Belong to the same molecular formula space as sitagliptin, i.e., they are isomers of C16H15F6N5O",
}
class MolecularPrompts:
    """Collection of prompts for molecular tasks"""
    
    # Mutation prompt template
    MUTATION_PROMPT = """You are a molecular designer tasked with creating chemical analogs.

Given the following molecule:
SMILES: {parent_smiles}

Please generate {num_mutations} chemical analogs by making small structural modifications.
Consider the following types of modifications:
- Add/remove functional groups
- Replace atoms (e.g., C->N, O->S)
- Change ring sizes
- Add/remove rings
- Modify side chains

Requirements:
1. Each analog should be chemically valid
2. Maintain similar molecular weight (±50 Da)
3. Keep modifications reasonable and drug-like

In your response, please provide a concise rationale summarizing how to edit the molecule. You need to conclude your answer with the sentence below (replacing the placeholder with the SMILES of your proposed molecule):\n\n
Based on the above analysis, the proposed molecule is: <box>Molecule SMILES</box>.
"""

    # Property optimization prompt
    OPTIMIZATION_PROMPT = """
    I have molecules and their {target_property} scores. {Property_description}

    {molecule_data}


    Please propose a new molecule that have a higher {target_property} score. You can edit the molecules above or propose a new one based on your knowledge.

    Requirements:
    - In your response, please provide a concise rationale summarizing how to achieve a high target score and then propose the molecule. You need to conclude your answer with the sentence below (replacing the placeholder with the SMILES of your proposed molecule):\n\n
    Based on the above analysis, the proposed molecule is: <box>Molecule SMILES</box>.
"""

    # Multi-objective optimization prompt
    MULTI_OBJECTIVE_PROMPT = """You are optimizing molecules for multiple properties simultaneously.

Current population:
{population_data}

Target properties and their importance:
{objectives}

Generate {num_molecules} new molecules that balance all objectives.
Consider trade-offs between properties and aim for Pareto-optimal solutions.

In your response, please provide a concise rationale summarizing how to achieve a high target score and then propose the molecule. You need to conclude your answer with the sentence below (replacing the placeholder with the SMILES of your proposed molecule):\n\n
Based on the above analysis, the proposed molecule is: <box>Molecule SMILES</box>.
"""

    # Analog generation prompt
    ANALOG_GENERATION_PROMPT = """You are an expert medicinal chemist. Generate {num_analogs} structural analogs of the following molecule that are optimized for {objective}.

Input molecule: {molecule}

Guidelines:
- Generate molecules with similar core structure but varied functional groups
- Ensure all molecules are valid SMILES strings
- Focus on modifications that would improve the objective
- Each analog should be chemically reasonable and synthetically accessible
- Output ONLY the SMILES strings, one per line, nothing else

Analogs:"""

    @staticmethod
    def get_mutation_prompt(parent_smiles: str, num_mutations: int = 5) -> Prompt:
        """Create mutation prompt"""
        return Prompt(
            custom_template=MolecularPrompts.MUTATION_PROMPT,
            default_vars={
                "parent_smiles": parent_smiles,
                "num_mutations": num_mutations
            }
        )
    
    @staticmethod
    def get_optimization_prompt(molecule_data: str, 
                              target_property: str) -> Prompt:
        """Create optimization prompt"""
        property_description = Property_description.get(target_property, "")
        return Prompt(
            custom_template=MolecularPrompts.OPTIMIZATION_PROMPT,
            default_vars={
                "molecule_data": molecule_data,
                "target_property": target_property,
                "Property_description": property_description
            }
        )
    
    @staticmethod
    def get_multi_objective_prompt(population_data: str,
                                  objectives: str,
                                  num_molecules: int = 10) -> Prompt:
        """Create multi-objective optimization prompt"""
        return Prompt(
            custom_template=MolecularPrompts.MULTI_OBJECTIVE_PROMPT,
            default_vars={
                "population_data": population_data,
                "objectives": objectives,
                "num_molecules": num_molecules
            }
        )