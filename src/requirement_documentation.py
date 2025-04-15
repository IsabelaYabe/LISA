"""
Criar métrica para calcular representatividade entre requerimento e usage_scenarios
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import setup_logger

logger = setup_logger()

class RequirementDocumentation:
    def __init__(self, id, filename, documentation_path, requirements):
        self.id = id
        self.filename = filename
        self.documentation_path = documentation_path
        self.requirements = requirements
    
    def add_requirement(self, requirement):
        logger.debug(f"Adding requirement: {requirement}")
        self.requirements.append(requirement)

    def remove_requirement(self, requirement):
        logger.debug(f"Removing requirement: {requirement}")
        self.requirements.remove(requirement)

    def get_requirement_by_id(self, id):
        logger.debug(f"Searching for requirement with ID: {id}")
        for req in self.requirements:
            if req.id == id:
                logger.debug(f"Requirement found: {req}")
                return req
        logger.debug(f"Requirement with ID {id} not found")
        return None

    def display_requirements(self):
        logger.debug("Displaying all requirements")
        for req in self.requirements:
            logger.debug(f"Requirement ID: {req.id}, Text: {req.text}")
            print(req)