"""
The kinds of background job LunaLab runs.

Only work that takes long enough to need it belongs here.  A single
analysis (one inference, a case study, a synthetic surface) takes well
under a second and is served directly by its own endpoint.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from backend.jobs import JobContext, JobType
from backend.services import INFERENCE_POSTERIOR_SAMPLES
from simulation.inference import Observation, infer

# Posterior summaries kept per crater: enough to tabulate and plot the batch
_BATCH_PARAMETERS = ("impactor_diameter", "velocity", "mass", "kinetic_energy")

CraterDiameter = Annotated[float, Field(gt=0, le=3_000_000)]


class InferenceBatchParams(BaseModel):
    """Invert many observed craters, each as POST /api/inference would."""
    crater_diameters: list[CraterDiameter] = Field(
        ..., min_length=1, max_length=500, description="Observed rim-to-rim diameters in metres"
    )
    diameter_error: float = Field(
        0.05, ge=0.01, le=1.0, description="Relative 1σ measurement error (0.05 = 5%)"
    )
    material: Optional[Literal["regolith", "rock"]] = Field(
        None, description="Target scaling; default: regolith for craters up to 125 m, rock above"
    )


def run_inference_batch(params: InferenceBatchParams, ctx: JobContext) -> dict:
    total = len(params.crater_diameters)
    craters = []
    ctx.progress(0.0)
    for done, diameter in enumerate(params.crater_diameters, start=1):
        posterior = infer(
            Observation(crater_diameter=diameter, diameter_error=params.diameter_error,
                        material=params.material),
            # Same as POST /api/inference, so a crater gets the same answer either
            # way; only the summaries are kept (the draws are 300 kB per crater)
            posterior_samples=INFERENCE_POSTERIOR_SAMPLES,
        )
        craters.append({
            "crater_diameter": diameter,
            "material": posterior["observation"]["material"],
            "effective_sample_size": posterior["effective_sample_size"],
            "posterior": {name: posterior["parameters"][name]["posterior"] for name in _BATCH_PARAMETERS},
        })
        ctx.progress(done / total)
    return {"count": total, "craters": craters}


JOB_TYPES: dict[str, JobType] = {
    "inference_batch": JobType(
        params_model=InferenceBatchParams,
        run=run_inference_batch,
        description="Infer the impactor behind each of up to 500 craters.",
    ),
}
