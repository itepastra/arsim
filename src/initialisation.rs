use ndarray::{Array, Axis};
use rand::{Rng, SeedableRng, rng};

use crate::{Error, Float, TwoDee};

/// returns `amount_of_particles` spaced around in a box with side lengths `box_size` in an fcc
/// lattice structure.
///
/// # Errors
///
/// This function will return an error if the amount of particles doesn't create a full crystal
/// lattice block.
pub(super) fn initial_positions(
    amount_of_particles: usize,
    box_size: Float,
) -> Result<TwoDee, Error> {
    let layers: usize = (amount_of_particles as Float / 4.0).powf(1.0 / 3.0).round() as usize;
    println!("there are {layers} layers");
    // TODO: make result type instead
    assert_eq!(layers.pow(3) * 4, amount_of_particles);

    // calculate lattice constant
    let lattice_constant = box_size / layers as Float;
    println!("the lattice constant is {lattice_constant}");
    let corner_offset = lattice_constant / 4.0;
    println!("the corner offset is {corner_offset}");

    let mut positions = Vec::with_capacity(amount_of_particles * 3);
    for x in 0..layers * 2 {
        for y in 0..layers * 2 {
            for z in 0..layers * 2 {
                if (x + y + z) % 2 == 0 {
                    positions.extend_from_slice(&vec![x as Float, y as Float, z as Float]);
                }
            }
        }
    }
    let mut positions = Array::from_shape_vec((amount_of_particles, 3), positions)?;
    positions *= lattice_constant / 2.0;
    positions += corner_offset;

    Ok(positions)
}

pub(super) fn initial_velocities(
    amount_of_particles: usize,
    temperature: Float,
    seed: Option<u64>,
) -> Result<TwoDee, Error> {
    let mut generator = match seed {
        Some(seed) => rand::rngs::SmallRng::seed_from_u64(seed),
        None => rand::rngs::SmallRng::try_from_os_rng()?,
    };

    let distribution = rand_distr::Normal::new(0.0, temperature.sqrt())?;
    let mut velocities = Array::from_shape_vec(
        (amount_of_particles, 3),
        (&mut generator)
            .sample_iter(distribution)
            .take(amount_of_particles * 3)
            .collect(),
    )?;
    assert_eq!(velocities.shape()[0], amount_of_particles);
    velocities -= &velocities
        .mean_axis(Axis(0))
        .expect("axis 0 is always in the velocities");
    Ok(velocities)
}
