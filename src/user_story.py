import regex as re
import spacy
from src.logger import setup_logger

logger = setup_logger()

class UserStory:
    def __init__(self, id, text):
        self.id = id
        self.text = text
        self.type_of_user, self.goal, self.reason = self.extract_user_story().values() 

    def extract_user_story(self):
        logger.debug("Starting the user story extraction process")
        pattern = r"As a (.*?), I want (.*?) so that (.*?)(?:\.|$)"
        match = re.findall(pattern, self.text, re.IGNORECASE)[0]
        result = {
                "type_of_user": match[0],
                "goal": match[1],
                "reason": match[2]
            }
        logger.debug(f"Extracted user story: {result}")
        return result