from astroquery.gaia import Gaia

query = """
SELECT
    source_id,
    ra, dec,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    ruwe
FROM gaiadr2.gaia_source
WHERE 1 = CONTAINS(
    POINT('ICRS', ra, dec),
    POLYGON(
      'ICRS',
      151.088008, 1.391339,
      149.407813, 1.391339,
      149.406853, 3.071038,
      151.088968, 3.071038
    )
  )
"""

job = Gaia.launch_job_async(query)
table = job.get_results()
table.write("output/gaia_dr2_cosmos.fits", overwrite=True)