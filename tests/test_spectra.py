"""Tests for generating spectra."""

import numpy as np


def test_integrated_generation_ngp(nebular_emission_model, random_part_stars):
    """Test the generation of integrated spectra."""
    # Compute the spectra using both the integrated and per particle machinery
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )
    per_particle_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra are different
    assert np.allclose(
        integrated_spec._lnu, per_particle_spec._lnu
    ), "The integrated and summed per particle spectra are not the same."


def test_integrated_generation_cic(nebular_emission_model, random_part_stars):
    """Test the generation of integrated spectra."""
    # Compute the spectra using both the integrated and per particle machinery
    nebular_emission_model.set_per_particle(False)
    integrated_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    random_part_stars.clear_all_emissions()
    nebular_emission_model.set_per_particle(True)
    per_particle_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )
    per_particle_spec = per_particle_spec.sum()

    # Ensure that the integrated spectra are different
    assert np.allclose(
        integrated_spec._lnu, per_particle_spec._lnu
    ), "The integrated and summed per particle spectra are not the same."


def test_threaded_generation_ngp_per_particle(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(True)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_threaded_generation_ngp_integrated(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(False)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_threaded_generation_cic_per_particle(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(True)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_threaded_generation_cic_integrated(
    nebular_emission_model,
    random_part_stars,
):
    """Test the generation of spectra with and without threading."""
    nebular_emission_model.set_per_particle(False)

    # Compute the spectra using both the integrated and per particle machinery
    serial_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=1,
    )
    random_part_stars.clear_all_emissions()
    threaded_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
        nthreads=4,
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        serial_spec._lnu, threaded_spec._lnu
    ), "The serial and threaded spectra are not the same."


def test_reusing_weights_ngp(nebular_emission_model, random_part_stars):
    """Test reusing weights to calculate another spectra for the same grid."""

    # Compute the spectra the first time
    first_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure we have the weights
    assert hasattr(
        random_part_stars, "_grid_weights"
    ), "The grid weights are not stored."
    assert (
        "test_grid" in random_part_stars._grid_weights["ngp"]
    ), "The grid weights are not stored."

    # Compute the spectra the second time which will reuse the weights
    random_part_stars.clear_all_emissions()
    second_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="ngp",
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        first_spec._lnu,
        second_spec._lnu,
    ), "The first and second spectra are not the same."


def test_reusing_weights_cic(nebular_emission_model, random_part_stars):
    """Test reusing weights to calculate another spectra for the same grid."""

    # Compute the spectra the first time
    first_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )

    # Ensure we have the weights
    assert hasattr(
        random_part_stars, "_grid_weights"
    ), "The grid weights are not stored."
    assert (
        "test_grid" in random_part_stars._grid_weights["cic"]
    ), "The grid weights are not stored."

    # Compute the spectra the second time which will reuse the weights
    random_part_stars.clear_all_emissions()
    second_spec = random_part_stars.get_spectra(
        nebular_emission_model,
        grid_assignment_method="cic",
    )

    # Ensure that the integrated spectra are different
    assert np.allclose(
        first_spec._lnu,
        second_spec._lnu,
    ), "The first and second spectra are not the same."
