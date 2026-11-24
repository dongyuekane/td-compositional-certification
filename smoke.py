"""Check one reduced solution and certificate against a full AC solve."""

from pathlib import Path
import sys


WORK = Path(__file__).resolve().parent / "work"
sys.path[:0] = [
    str(WORK / "research/control_blocks"),
    str(WORK / "research/normalized_response"),
]

from atlas import Atlas, Reduced, SEED, reference
from coupled_certificate import certify_main, slack_loss_interval
from feeder_tube import certify_response


def main():
    plan = dict(T=0.0, X=0.0, F=0, S=0, R=1, C=1)
    control = next(reference.controls(plan))
    system = reference.build("mixed9", 0.9, plan, control, shape_seed=SEED)
    model = Reduced(system, Atlas(degree=6))
    reduced = model.solve()
    tubes = [certify_response(response) for response in model.responses]
    certificate = certify_main(model, reduced, tubes)
    exact = system.solve()
    assert exact["converged"]
    assert certificate["certified"]
    loss_lower, loss_upper = slack_loss_interval(model, reduced, certificate)
    assert loss_lower - 1e-9 <= exact["loss"] <= loss_upper + 1e-9
    print(f"AC loss {exact['loss']:.8f} lies in [{loss_lower:.8f}, {loss_upper:.8f}]")


if __name__ == "__main__":
    main()
