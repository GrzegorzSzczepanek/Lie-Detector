import os  
from typing import Optional 
 
import brainaccess.core as bacore 
from brainaccess.core.eeg_manager import EEGManager 
from brainaccess.utils import acquisition 
 
from ..common import ExperimentBlock, ParticipantResponse 
from ..config import PORT 
from ..personal_data import PersonalDataField, PersonalDataType 
from .eeg_config import DATA_FOLDER_PATH, USED_DEVICE 
 
 
class EEGHeadset: 
    def __init__(self, participant_id: str) -> None: 
        """ 
        Args: 
            participant_id (str): ID to use as folder name for saved data. 
        """ 
 
        self._is_started: bool = False 
        self._is_connected: bool = False 
 
        self._current_block: Optional[ExperimentBlock] = None 
        self._is_block_started: bool = False 
        self._save_dir_path: str = os.path.join(DATA_FOLDER_PATH, participant_id) 
        # inicjalizacja biblioteki  
        bacore.init(bacore.Version(2, 0, 0))  
        self._create_dir_if_not_exist(DATA_FOLDER_PATH)  
        self._create_dir_if_not_exist(self._save_dir_path)  
 
    def _connect(self) -> None: 
        """ 
        Raises: 
            RuntimeError: If cannot connect to the headset. 
        """ 
 
        self._eeg_manager = EEGManager() 
        self._eeg_acquisition = acquisition.EEG() 
        # łączenie się z opaską 
        self._eeg_acquisition.setup(self._eeg_manager, USED_DEVICE, port=PORT) 
        self._is_connected = True   
 
    def _disconnect(self) -> None: 
        """ 
        Raises: 
            RuntimeError: If there is no connection to the headset or cannot disconnect. 
        """ 
 
        if self._is_block_started: 
            raise RuntimeError("Another block is already started.") 
 
        if not self._is_connected: 
            raise RuntimeError("The EEG headset is not connected.") 
 
        self._eeg_manager.disconnect() 
 
    def start_block(self, experiment_block: ExperimentBlock) -> None: 
        """ 
        Args: 
            experiment_block (ExperimentBlock): Experiment block to register data for. 
        """ 
 
        if self._is_block_started: 
            raise RuntimeError("Another block is already started.") 
 
        self._connect() 
        self._experiment_block = experiment_block 
        # Rozpoczęcie akwizycji 
        self._eeg_acquisition.start_acquisition() 
        self._is_started = True 
        self._current_block = experiment_block 
 
    def stop_and_save_block(self) -> None: 
        """ 
        Raises: 
            RuntimeError: If block was not started. 
        """ 
 
        # Pobranie danych EEG 
        self._eeg_acquisition.get_mne() 
 
        # Zapisanie danych do pliku 
        file_path = os.path.join( 
            self._save_dir_path, f"EEG_{self._current_block}_raw.fif" 
        ) 
        self._eeg_acquisition.data.save(file_path) 
 
        self._is_started = False 
 
        # przerwanie akwizycji 
        self._eeg_acquisition.stop_acquisition() 
        self._eeg_manager.clear_annotations() 
        self._disconnect() 
 
    def annotate_data_shown( 
        self, shown_data_field: PersonalDataField, shown_data_type: PersonalDataType 
    ) -> None: 
        """ 
        Args: 
            shown_data_field (PersonalDataField): Exact field (name, birth date...) of data shown to the participant. 
            shown_data_type (PersonalDataType): Type of personal data shown to the participant (real, fake...). 
 
        Raises: 
            RuntimeError: If block was not started. 
        """ 
 
        if not self._is_connected: 
            raise RuntimeError("The headset is not connected.") 
 
        if self._is_block_started: 
            raise RuntimeError("Another block is already started.") 
 
        data_to_annotate = f"{shown_data_field}: {shown_data_type}" 
        self._eeg_acquisition.annotate(data_to_annotate) 
 
    def annotate_response(self, participant_response: ParticipantResponse) -> None: 
        """ 
            Args:class EEGHeadset: 
        def __init__(self, participant_id: str) -> None: 
        """ 
        if not self._is_connected: 
            raise RuntimeError("The headset is not connected.") 
        # Tworzenie adnotacji na podstawie odpowiedzi uczestnika 
        self._eeg_acquisition.annotate(str(participant_response)) 
 
    def _create_dir_if_not_exist(self, path: str) -> None: 
        if not os.path.exists(path): 
            os.mkdir(path) 
 