"""Compatibility imports for shared temporary-service continuation contracts."""

from cayu.sessions._temporary_continuation import (
    ServiceReleasedSessionStatus as ServiceReleasedSessionStatus,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission as TemporaryServiceAdmission,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceDispatch as TemporaryServiceDispatch,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceExclusion as TemporaryServiceExclusion,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceExecution as TemporaryServiceExecution,
)
from cayu.sessions._temporary_continuation import TemporaryServiceIntent as TemporaryServiceIntent
from cayu.sessions._temporary_continuation import (
    TemporaryServicePreparation as TemporaryServicePreparation,
)
from cayu.sessions._temporary_continuation import TemporaryServiceRecord as TemporaryServiceRecord
from cayu.sessions._temporary_continuation import (
    advance_temporary_service_record as advance_temporary_service_record,
)
from cayu.sessions._temporary_continuation import reference_for_service as reference_for_service
from cayu.sessions._temporary_continuation import (
    require_temporary_service_capacity as require_temporary_service_capacity,
)
from cayu.sessions._temporary_continuation import (
    require_temporary_service_command as require_temporary_service_command,
)
from cayu.sessions._temporary_continuation import (
    temporary_admission_payload_sha256 as temporary_admission_payload_sha256,
)
from cayu.sessions._temporary_continuation import (
    temporary_service_invocation_id as temporary_service_invocation_id,
)
from cayu.sessions._temporary_continuation import temporary_service_key as temporary_service_key
