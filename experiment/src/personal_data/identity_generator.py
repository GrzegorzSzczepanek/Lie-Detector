from pathlib import Path

import yaml

from .config import (
    CELEBRITIES_FILE_PATH,
    FAKE_AND_RANDO_DATA_FILE_PATH,
    USER_DATA_FILE_PATH,
)
from .personal_data import PersonalDataField


class IdentityGenerator:
    def __init__(self):
        self._temp_user = self._load_user_data()
        self._identity_data = self._load_identity_data()
        

        # Filter user data right away
        self._filter_identity_data(self._temp_user)

        self._celebrities = self._load_celebrities_data()["celebrities"]

        fake = self._create_identity()
        rando = self._create_identity()

        self._concat_name_and_surname(self._temp_user)
        self._concat_name_and_surname(fake)
        self._concat_name_and_surname(rando)

        self._celebrity = self._str_keys_to_enum(self._pick_celebrity())
        self._fake = self._str_keys_to_enum(fake)
        self._rando = self._str_keys_to_enum(rando)

        self._id = self._temp_user.pop("id")
        self._temp_user.pop("gender", None)
        
        # self.remove_parents_names()
        self._user = self._str_keys_to_enum(self._temp_user)
        

    def _concat_name_and_surname(self, person: dict[str, str]) -> None:
        person["name"] = f"{person['name']} {person['surname']}"
        del person["surname"]
    
    # def remove_parents_names(self):
    #     if 'fathers_name' in self._temp_user:
    #         del self._temp_user['fathers_name']
    #     if 'mothers_name' in self._temp_user:
    #         del self._temp_user['mothers_name']

    def _str_keys_to_enum(
        self, person_data: dict[str, str]
    ) -> dict[PersonalDataField, str]:
        return {
            PersonalDataField.from_string(key): value
            for key, value in person_data.items()
        }

    def get_real(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: Participant identity data.
        """

        return self._user

    def get_fake(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: Fake identity data.
        """

        return self._fake

    def get_celebrity(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: Celebrity identity data.
        """

        return self._celebrity

    def get_rando(self) -> dict[PersonalDataField, str]:
        """
        Returns:
            dict[PersonalDataField, str]: Rando identity data.
        """

        return self._rando

    def get_id(self) -> str:
        """
        Returns:
            str: Participant's ID.
        """

        return self._id

    def _validate_identity_data(self, identity_data: dict[str, list[str]]) -> None:
        """
        Raises:
            ValueError: If the identity data is invalid.
        """
        if identity_data is None:
            raise ValueError("Identity data cannot be None.")
        elif not isinstance(identity_data, dict):
            raise ValueError(
                "Identity data must be a dictionary. Got: {type(identity_data)}"
            )
        elif not all(
            key in identity_data
            for key in ["female_names", "male_names", "hometowns", "birth_dates", "surnames"]
        ):
            raise ValueError(
                f"Identity data must contain 'female_names', 'male_names', 'hometowns', 'birth_dates', 'surnames' keys. Got: {self._identity_data.keys()}"
            )
        elif not all(len(value) > 0 for value in identity_data.values()):
            raise ValueError("Identity data values must not be empty.")
        elif not all(isinstance(value, list) for value in identity_data.values()):
            raise ValueError("Identity data values must be list of strings.")
        elif not all(
            isinstance(item, str)
            for sublist in identity_data.values()
            for item in sublist
        ):
            raise ValueError("Identity data values must be lists of strings.")

    def _validate_person_data(self, identity_data: dict[str, str]) -> None:
        """
        Args:
            identity_data (dict[str, str]): Identity data to be validated.
        Raises:
            ValueError: If the identity data is invalid.
        """
        if identity_data is None:
            raise ValueError("Identity data cannot be None.")
        elif not isinstance(identity_data, dict):
            raise ValueError("Identity data must be a dictionary.")
        elif not all(
            key in identity_data
            for key in [
                "name",
                "surname",
                "gender",
                "hometown",
                "birth_date",
            ]
        ):
            raise ValueError(
                f"Identity data must contain 'name', 'surname', 'hometown', 'gender', 'birth_date' keys. \nGot: {identity_data}"
            )
        elif not all(
            value is not None and len(value) > 0 for value in identity_data.values()
        ):
            raise ValueError("Identity data values must not be empty.")
        elif not all(isinstance(value, str) for value in identity_data.values()):
            raise ValueError("Identity data values must be strings.")
        elif identity_data["gender"] not in ["m", "k"]:
            raise ValueError("Identity data gender must be 'm' or 'k'.")

    def _load_yaml(self, file_path: Path) -> dict:
        """
        Args:
            file_path (Path): Path to the YAML file.
        Returns:
            dict: Loaded data from the YAML file.
        Raises:
            FileNotFoundError: If the YAML file was not found.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)

        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} was not found.")

    def _load_identity_data(self) -> dict[str, list[str]]:
        """
        Returns:
            dict[str, list[str]]: Loaded data from the YAML file.
        Raises:
            FileNotFoundError: If the YAML file was not found.
        """
        identity_data = self._load_yaml(FAKE_AND_RANDO_DATA_FILE_PATH)
        self._validate_identity_data(identity_data)
        return identity_data

    def _load_user_data(self) -> dict[str, str]:
        """
        Load user data from a YAML file.

        Returns:
            dict: Loaded user data.
        """
        user_data = self._load_yaml(USER_DATA_FILE_PATH)
        self._validate_person_data(user_data)
        return user_data

    def _load_celebrities_data(self) -> dict:
        """
        Load celebrities data from a YAML file.

        Returns:
            dict: Loaded celebrities data.
        Raises:
            FileNotFoundError: If the celebrities data YAML file was not found.
        """
        celebrities_data = self._load_yaml(CELEBRITIES_FILE_PATH)
        return celebrities_data

    def _pick_celebrity(self) -> dict[str, str]:
        """
        Returns first celebrity that has no data in common with user
        """
        user_data_list = [
            self._load_user_data().get(key, "")
            for key in [
                "name",
                "surname",
                "hometown",
                "birth_date",
            ]
        ]
        return [
            celebrity
            for celebrity in self._celebrities
            if not any(
                data
                in [
                    celebrity.get("name", "").split()[0],
                    celebrity.get("name", "").split()[1],
                    celebrity.get("hometown", ""),
                    celebrity.get("birth_date", ""),
                ]
                for data in user_data_list
            )
        ][0]

    def _create_identity(self) -> dict[str, str]:
        """
        Returns randomly created person that has no data in common with user and celebrity
        """
        identity = {}
        if self._temp_user["gender"] == "m":
            identity["name"] = self._identity_data["male_names"][0]
        else:
            identity["name"] = self._identity_data["female_names"][0]

        identity["surname"] = self._identity_data["surnames"][0]
        identity["hometown"] = self._identity_data["hometowns"][0]
        identity["birth_date"] = self._identity_data["birth_dates"][0]  # Use the correct key here

        self._filter_identity_data(identity)
        return identity

    def _filter_identity_data(self, person: dict[str, str]) -> None:
        for category, data in self._identity_data.items():
            match category:
                case "female_names":
                    data_not_in_common = [
                        name
                        for name in data
                        if name != person["name"]
                    ]
                    self._identity_data[category] = data_not_in_common

                case "male_names":
                    data_not_in_common = [
                        name
                        for name in data
                        if name != person["name"]
                    ]
                    self._identity_data[category] = data_not_in_common

                case "hometowns":
                    data_not_in_common = [
                        hometown for hometown in data if hometown != person["hometown"]
                    ]
                    self._identity_data[category] = data_not_in_common

                case "birth_dates":
                    data_not_in_common = [
                        birth_date for birth_date in data if birth_date != person["birth_date"]
                    ]
                    self._identity_data[category] = data_not_in_common

                case "surnames":
                    data_not_in_common = [
                        surname for surname in data if surname != person["surname"]
                    ]
                    self._identity_data[category] = data_not_in_common
