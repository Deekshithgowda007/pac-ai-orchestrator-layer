from pynetdicom import AE
from pynetdicom.sop_class import ComprehensiveSRStorage


def send_sr_to_dcm4chee(ds):
    ae = AE(ae_title="AI_WORKER")

    ae.add_requested_context(ComprehensiveSRStorage)

    assoc = ae.associate(
        addr="dcm4chee-arc",
        port=11112,
        ae_title="DCM4CHEE"
    )

    if not assoc.is_established:
        raise RuntimeError("❌ Could not associate with dcm4chee")

    status = assoc.send_c_store(ds)
    assoc.release()

    if status:
        return True
    else:
        raise RuntimeError("❌ C-STORE failed")
