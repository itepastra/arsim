use std::f64::INFINITY;

use ndarray::{AssignElem, Axis, stack};

use crate::{Error, Float, ThreeDee, TwoDee};

pub fn lj_force(relative_positions: ThreeDee, distances: &TwoDee) -> (TwoDee, TwoDee) {
    let force_magnitude = 24.0 * distances.powi(-7) - 48.0 * distances.powi(-13);
    assert!(!force_magnitude.is_any_nan());
    let force_direction = relative_positions / distances.view().insert_axis(Axis(2));
    assert!(!force_direction.is_any_nan());
    let force_matrix = force_direction * force_magnitude.view().insert_axis(Axis(2));
    let net_force = -force_matrix.sum_axis(Axis(0));
    assert_eq!(net_force.shape()[1], 3);
    (force_magnitude, net_force)
}

pub fn atomic_distances(
    initial_positions: &TwoDee,
    box_dim: Float,
) -> Result<(ThreeDee, TwoDee), Error> {
    let amount_of_particles = initial_positions.shape()[0];
    let dists: Vec<_> = initial_positions
        .columns()
        .into_iter()
        .map(|col| {
            (&col
                .to_shape((1, amount_of_particles))
                .expect("column doesn't have amount_of_particles entries")
                - &col
                    .to_shape((amount_of_particles, 1))
                    .expect("column doesn't have amount_of_particles entries")
                + box_dim * 0.5)
                % box_dim
                - box_dim * 0.5
        })
        .collect();
    assert_eq!(dists.len(), 3);
    let relative_positions = stack!(Axis(2), dists[0], dists[1], dists[2]);
    let mut distances =
        (&dists[0] * &dists[0] + &dists[1] * &dists[1] + &dists[2] * &dists[2]).sqrt();
    distances.diag_mut().fill(INFINITY as Float);
    assert!(!relative_positions.is_any_nan());
    assert_eq!(relative_positions.shape()[0], amount_of_particles);
    assert_eq!(relative_positions.shape()[1], amount_of_particles);
    assert_eq!(relative_positions.shape()[2], 3);
    assert!(!distances.is_any_nan());
    assert_eq!(distances.shape()[0], amount_of_particles);
    assert_eq!(distances.shape()[1], amount_of_particles);
    Ok((relative_positions, distances))
}

pub fn kinetic_energy(velocities: &TwoDee) -> Float {
    0.5 * velocities.powi(2).sum()
}

pub fn potential_energy(distances: &TwoDee) -> Float {
    let individual = 4.0 * (distances.powi(-12) - distances.powi(-6));
    0.5 * individual.sum()
}

pub fn temperature(kinetic_energy: Float, amount_of_particles: usize) -> Float {
    2.0 * kinetic_energy / (3.0 * (amount_of_particles - 1) as Float)
}

#[cfg(test)]
mod test {
    use std::f64::{INFINITY, consts::SQRT_2};

    use ndarray::{Array, array};

    use crate::Float;

    use super::{atomic_distances, lj_force};

    #[test]
    fn test_relative_positions() {
        let positions = array![[1.0, 2.0, 3.0], [0.0, 2.0, 2.0]];
        let (relative_positions, distances) = atomic_distances(&positions, 5.0).unwrap();
        assert_eq!(
            relative_positions.flatten_with_order(ndarray::Order::RowMajor),
            array![
                [[0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]],
                [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
            ]
            .flatten_with_order(ndarray::Order::RowMajor)
        );
        assert_eq!(
            distances,
            array![
                [INFINITY as Float, SQRT_2 as Float],
                [SQRT_2 as Float, INFINITY as Float]
            ]
        );
    }

    #[test]
    fn test_force_direction() {
        let positions = array![[1.0, 2.0, 3.0], [0.0, 2.0, 3.0]];
        let (relative_positions, distances) = atomic_distances(&positions, 50.0).unwrap();
        let (magnitude, force) = lj_force(relative_positions, &distances);
        println!("{:#?}", force);
        println!("{:#?}", magnitude);
        todo!()
    }
}
