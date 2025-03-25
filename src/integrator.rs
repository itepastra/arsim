use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Axis, Dimension};

use crate::{
    Error, Float, OneDee, ThreeDee, TwoDee,
    physics::{atomic_distances, kinetic_energy, lj_force, potential_energy, temperature},
};

pub struct IntegrationStepResult {
    positions: TwoDee,
    velocities: TwoDee,
    forces: TwoDee,
    force_magnitudes: TwoDee,
    distances: TwoDee,
}

#[derive(Debug)]
pub struct IntegrationResult {
    positions: ThreeDee,
    velocities: ThreeDee,
    virials: OneDee,
    kinetic_energies: OneDee,
    potential_energies: OneDee,
}

pub trait Integrator {
    fn initialisation(&mut self) -> Result<(), Error>;
    fn deinit(&mut self) -> Result<(), Error>;
    fn integration_step(
        &mut self,
        positions: TwoDee,
        velocities: TwoDee,
        forces: TwoDee,
        time_step_size: Float,
        box_dim: Float,
    ) -> Result<IntegrationStepResult, Error>;
    fn simulate(
        &mut self,
        initial_positions: TwoDee,
        initial_velocities: TwoDee,
        time_step_size: Float,
        max_time: Float,
        box_dim: Float,
    ) -> Result<IntegrationResult, Error> {
        assert_eq!(
            initial_positions.shape()[0],
            initial_velocities.shape()[0],
            "positions and velocities contain a different amount of particles"
        );
        assert!(
            time_step_size < max_time,
            "time step size larger then max time"
        );
        let amount_of_particles = initial_positions.shape()[0];
        let timesteps = (max_time / time_step_size) as usize;
        let r_max = (box_dim.powi(2) * 3.0).sqrt();

        let mut positions = Array::zeros((amount_of_particles, 3, timesteps));
        let mut velocities = Array::zeros((amount_of_particles, 3, timesteps));

        let mut kinetic_energies = Array::zeros(timesteps);
        let mut potential_energies = Array::zeros(timesteps);
        let mut virials = Array::zeros(timesteps);
        let mut temperatures = Array::zeros(timesteps);
        let mut equilibrium_timestep: Option<usize> = None;

        self.initialisation()?;
        let mut current_positions = initial_positions;
        let mut current_velocities = initial_velocities;
        let (mut relative_positions, mut distances) =
            atomic_distances(&current_positions, box_dim)?;
        let (mut force_magnitudes, mut forces) = lj_force(relative_positions, &distances);
        let bar = ProgressBar::new(timesteps as u64).with_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}/{duration_precise} (Remaining: {eta_precise})] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        for step in 0..timesteps {
            // start by assigning all the stuff
            positions
                .index_axis_mut(Axis(2), step)
                .assign(&current_positions);
            velocities
                .index_axis_mut(Axis(2), step)
                .assign(&current_velocities);

            kinetic_energies[step] = kinetic_energy(&current_positions);
            potential_energies[step] = potential_energy(&distances);
            virials[step] = 0.5 * (distances * force_magnitudes).sum();
            temperatures[step] = temperature(kinetic_energies[step], amount_of_particles);

            // I can move them here, this will give new ones
            let IntegrationStepResult {
                positions: new_positions,
                velocities: new_velocities,
                forces: new_forces,
                force_magnitudes: new_magnitudes,
                distances: new_distances,
            } = self.integration_step(
                current_positions,
                current_velocities,
                forces,
                time_step_size,
                box_dim,
            )?;

            bar.inc(1);
            current_positions = new_positions;
            current_velocities = new_velocities;
            forces = new_forces;
            force_magnitudes = new_magnitudes;
            distances = new_distances;
        }
        bar.finish_with_message("Finished simulation");
        Ok(IntegrationResult {
            positions,
            velocities,
            virials,
            kinetic_energies,
            potential_energies,
        })
    }
}

pub struct Verlet {}

impl Integrator for Verlet {
    fn initialisation(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn deinit(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn integration_step(
        &mut self,
        positions: TwoDee,
        velocities: TwoDee,
        forces: TwoDee,
        time_step_size: Float,
        box_dim: Float,
    ) -> Result<IntegrationStepResult, Error> {
        let new_positions = positions
            + &velocities * time_step_size
            + &forces * time_step_size * time_step_size * 0.5;
        let (relative_positions, new_distances) = atomic_distances(&new_positions, box_dim)?;
        let (new_magnitudes, new_forces) = lj_force(relative_positions, &new_distances);
        let dvel = (forces + &new_forces) * time_step_size * 0.5;
        let new_velocities = velocities + dvel;

        Ok(IntegrationStepResult {
            positions: new_positions,
            velocities: new_velocities,
            forces: new_forces,
            force_magnitudes: new_magnitudes,
            distances: new_distances,
        })
    }
}

pub struct VerletCUDA {}

impl Integrator for VerletCUDA {
    fn initialisation(&mut self) -> Result<(), Error> {
        todo!("implement CUDA integrator");
        Ok(())
    }

    fn deinit(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn integration_step(
        &mut self,
        positions: TwoDee,
        velocities: TwoDee,
        forces: TwoDee,
        time_step_size: Float,
        box_dim: Float,
    ) -> Result<IntegrationStepResult, Error> {
        let new_positions = positions
            + &velocities * time_step_size
            + &forces * time_step_size * time_step_size * 0.5;
        let (relative_positions, new_distances) = atomic_distances(&new_positions, box_dim)?;
        let (new_magnitudes, new_forces) = lj_force(relative_positions, &new_distances);
        let dvel = (forces + &new_forces) * time_step_size * 0.5;
        let new_velocities = velocities + dvel;

        Ok(IntegrationStepResult {
            positions: new_positions,
            velocities: new_velocities,
            forces: new_forces,
            force_magnitudes: new_magnitudes,
            distances: new_distances,
        })
    }
}
