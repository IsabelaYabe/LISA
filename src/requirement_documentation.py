
class RequirementDocumentation:
    def __init__(self, id, name, documentation, requirements):
        self.requirements = []

    def add_requirement(self, requirement):
        """
        Add a new requirement to the documentation.

        :param requirement: The requirement to be added.
        """
        self.requirements.append(requirement)

    def display_requirements(self):
        """
        Display all requirements in the documentation.
        """
        for req in self.requirements:
            print(req)