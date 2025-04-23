import sys
import os
from dataclasses import dataclass, field   
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from logger import logger

@dataclass(frozen=True, slots=True)
class UserStory:
        id: str
        text: str
        type_of_user: str 
        goal: str 
        reason: str 
        requirement: str
        doc: str
        metadata: str
        verbs: list[str] = field(default_factory=list)
        auxs: list[str] = field(default_factory=list)
        nouns: list[str] = field(default_factory=list) 
        propns: list[str] = field(default_factory=list)
        prons: list[str] = field(default_factory=list) 
        adjs: list[str] = field(default_factory=list) 
        advs: list[str] = field(default_factory=list)
        sconjs: list[str] = field(default_factory=list) 
        parts: list[str] = field(default_factory=list) 
        orgs: list[str] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class UsageScenario:
        id: str
        text: str
        requirements: str
        verbs: list[str] = field(default_factory=list)
        auxs: list[str] = field(default_factory=list)
        nouns: list[str] = field(default_factory=list) 
        propns: list[str] = field(default_factory=list)
        prons: list[str] = field(default_factory=list) 
        adjs: list[str] = field(default_factory=list) 
        advs: list[str] = field(default_factory=list)
        sconjs: list[str] = field(default_factory=list) 
        parts: list[str] = field(default_factory=list) 
        orgs: list[str] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class Requirement:
        id: str 
        key: str
        text: str
        metadata: str
        user_stories: list[str] = field(default_factory=list)
        usage_scenarios: list[str] = field(default_factory=list)
        verbs: list[str] = field(default_factory=list)
        auxs: list[str] = field(default_factory=list)
        nouns: list[str] = field(default_factory=list) 
        propns: list[str] = field(default_factory=list)
        prons: list[str] = field(default_factory=list) 
        adjs: list[str] = field(default_factory=list) 
        advs: list[str] = field(default_factory=list)
        sconjs: list[str] = field(default_factory=list) 
        parts: list[str] = field(default_factory=list) 
        orgs: list[str] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class RequirementDocumentation:
        id: str
        title: str
        text: str
        documentation_path: str
        metadata: str
        requirements: list[Requirement] = field(default_factory=list)
        user_stories: list[UserStory] = field(default_factory=list)
        usage_scenarios: list[UsageScenario] = field(default_factory=list)