import os
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True, slots=True)
class RequirementDocumentationID:
    id: int
    
    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return f"RequirementDocumentationID(id={self.id})"

    def __eq__(self, other):
        return isinstance(other, RequirementDocumentationID) and self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
@dataclass(frozen=True, slots=True)
class MetadataID:
    requirement_documentation_id: RequirementDocumentationID
    metadata_key: int
    
    def __str__(self):
        return f"{self.requirement_documentation_id}-{self.metadata_key}"

    def __repr__(self):
        return f"MetadataID(requirement_documentation_id={self.requirement_documentation_id}, metadata_key={self.metadata_key})"
    
    def __eq__(self, other):
        return isinstance(other, MetadataID) and self.requirement_documentation_id == other.requirement_documentation_id and self.metadata_key == other.metadata_key

    def __hash__(self):
        return hash((self.requirement_documentation_id, self.metadata_key))
    
@dataclass(frozen=True, slots=True)        
class RequirementID:
    requirement_documentation_id: RequirementDocumentationID
    metadata_key: int
    requirement_key: int

    def __str__(self):
        return f"{self.requirement_documentation_id}-{self.metadata_key}-{self.requirement_key}"

    def __repr__(self):
        return f"RequirementID(requirement_documentation_id={self.requirement_documentation_id}, metadata_key={self.metadata_key}, requirement_key={self.requirement_key})"

    def __eq__(self, other):
        return isinstance(other, RequirementID) and self.requirement_documentation_id == other.requirement_documentation_id and self.metadata_key == other.metadata_key and self.requirement_key == other.requirement_key

    def __hash__(self):
        return hash((self.requirement_documentation_id, self.metadata_key, self.requirement_key))


@dataclass(frozen=True, slots=True)    
class UserStoryID:
    requirement_id: RequirementID
    user_story_requirement_id: int
    index: int
    
    def __str__(self):
        return f"{self.requirement_id}-{self.user_story_requirement_id}.{self.index}"
    
    def __repr__(self):
        return f"UserStoryID(requirement_id={self.requirement_id}, user_story_requirement_id={self.user_story_requirement_id}, index={self.index})"
    
    def __eq__(self, other):
        return isinstance(other, UserStoryID) and self.requirement_id == other.requirement_id and self.user_story_requirement_id == other.user_story_requirement_id and self.index == other.index
    
    def __hash__(self):
        return hash((self.requirement_id, self.user_story_requirement_id, self.index))

@dataclass(frozen=True, slots=True)
class UsageScenarioID:
    requirement_documentation_id: RequirementDocumentationID
    cluster_id: int

    def __str__(self):
        return f"{self.requirement_documentation_id}-cluster{self.cluster_id}"

    def __repr__(self):
        return f"UsageScenarioID(requirement_documentation_id={self.requirement_documentation_id}, cluster_id={self.cluster_id})"

    def __eq__(self, other):
        return isinstance(other, UsageScenarioID) and self.requirement_documentation_id == other.requirement_documentation_id and self.cluster_id == other.cluster_id

    def __hash__(self):
        return hash((self.requirement_documentation_id, self.cluster_id))

class IDGenerator:
    def __init__(self, requirement_docs_folder_path: str):
        self._filenames_map, self._requirement_docs_id = self.__generate_filename_map(requirement_docs_folder_path)

    def __generate_filename_map(self, directory):
        filenames_map = {}
        map_filenames = {}

        files = sorted(os.listdir(directory))
        for i, file in enumerate(files):
            filename = os.path.splitext(file)[0]
            req_doc_id = RequirementDocumentationID(id=i)
            filenames_map[filename] = req_doc_id
            map_filenames[req_doc_id] = filename

        return filenames_map, map_filenames 
    
    def generate_requirement_documentation_id(self, filename: str) -> RequirementDocumentationID:
        return self.filenames_map[filename]
    
    def generate_metadata_id(self, filename: str, metadata_key: int) -> MetadataID:
        req_doc_id = self.generate_requirement_documentation_id(filename)
        return MetadataID(requirement_documentation_id=req_doc_id, metadata_key=metadata_key)
    
    def generate_requirement_id(self, filename: str, metadata_key: int, requirement_key: int) -> RequirementID:
        req_doc_id = self.generate_requirement_documentation_id(filename)
        return RequirementID(requirement_documentation_id=req_doc_id, metadata_key=metadata_key, requirement_key=requirement_key)
    
    def generate_user_story_id(self, filename: str, metadata_key: int, requirement_key: int, user_story_requirement_id: int, index: int) -> UserStoryID:
        req_id = self.generate_requirement_id(filename, metadata_key, requirement_key)
        return UserStoryID(requirement_id=req_id, user_story_requirement_id=user_story_requirement_id, index=index)
    
    def generate_usage_scenario_id(self, filename: str, cluster_id: int) -> UsageScenarioID:
        req_doc_id = self.generate_requirement_documentation_id(filename)
        return UsageScenarioID(requirement_documentation_id=req_doc_id, cluster_id=cluster_id)
