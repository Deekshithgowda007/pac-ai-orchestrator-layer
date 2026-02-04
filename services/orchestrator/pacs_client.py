import requests
from requests.auth import HTTPBasicAuth

class PACSClient:
    def __init__(self, base, aet, user, pwd):
        self.base = base
        self.aet = aet
        self.auth = HTTPBasicAuth(user, pwd)

    def get_instances(self, study_uid):
        url = f"{self.base}/aets/{self.aet}/rs/studies/{study_uid}/instances"
        r = requests.get(url, headers={"Accept":"application/dicom+json"}, auth=self.auth)
        r.raise_for_status()
        return r.json()
