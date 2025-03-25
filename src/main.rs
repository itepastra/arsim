use integrator::{Integrator, Verlet, VerletCUDA};
use ndarray::{ArrayBase, Dim, OwnedRepr};

mod initialisation;
mod integrator;
mod physics;

type OneDee = ArrayBase<OwnedRepr<Float>, Dim<[usize; 1]>>;
type TwoDee = ArrayBase<OwnedRepr<Float>, Dim<[usize; 2]>>;
type ThreeDee = ArrayBase<OwnedRepr<Float>, Dim<[usize; 3]>>;
type Error = Box<dyn std::error::Error>;
type Float = f32;

const SEED: Option<u64> = Some(33);
const BOX_SIZE: Float = 8.0;
const NUMBER_OF_PARTICLES: usize = 500;
const TIME_STEP: Float = 0.005;
const TOTAL_TIME: Float = 10.0;

fn main() -> Result<(), Error> {
    // parse configuration
    let positions = initialisation::initial_positions(NUMBER_OF_PARTICLES, BOX_SIZE)?;
    let velocities = initialisation::initial_velocities(NUMBER_OF_PARTICLES, 0.01, SEED)?;

    let mut integrator = Verlet {};
    let result = integrator.simulate(positions, velocities, TIME_STEP, TOTAL_TIME, BOX_SIZE)?;
    println!("{:#?}", result);

    Ok(())
}
