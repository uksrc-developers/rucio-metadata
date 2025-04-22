"""
Generate a companion metadata file for uploading data via Rucio.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from astropy.io import fits
from astropy.coordinates import Galactic, ICRS
from astropy import units as u
import json
import numpy as np
from pathlib import Path

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "file_paths",
        type=str,
        nargs="+",
        help="Path to the input file(s).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="testing",
        help="Namespace for Rucio.  Defaults to 'testing'.",
    )
    parser.add_argument(
        "--lifetime",
        type=int,
        default=86400,
        help="Lifetime of the data product in seconds.  Defaults to 86400.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="Output directory for metadata files.  Defaults to './'."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output."
    )
    return parser.parse_args()

def get_fits_header_dict(fpath, verbose=False, indent="\t") -> dict:
    """
    Get required info from a FITS header.

    Parameters
    ----------
    fpath : Path or str
        Path to a FITS file.
    verbose : bool, optional
        Print verbose output.  Defaults to False.
    indent : str, optional
        Indentation for verbose output.  Defaults to '\t'.

    Returns
    -------
    header_dict : dict
        Dictionary containing relevant header information.
    """
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist.")

    header_dict = {}
    with fits.open(fpath) as hdul:
        if len(hdul) > 1:
            if verbose:
                print(
                    indent
                    + f"More than one HDU found.  "
                    + f"Trying to find the relevant HDU... ",
                    end=""
                )
            # Multiple header data units
            # Try and find the header with the details we want
            naxis_vals = []
            for i in range(len(hdul)):
                if "NAXIS" in hdul[i].header:
                    naxis_vals.append(hdul[i].header["NAXIS"])
                else:
                    naxis_vals.append(0)
            header_ind = np.where(np.array(naxis_vals) > 0)[0]
            assert header_ind.size > 0, "No valid header found."
            header_ind = header_ind[0]
            if verbose:
                print(f"(HDU index {header_ind})")
        else:
            header_ind = 0
        header = hdul[header_ind].header

        if "PIXTYPE" in header:
            pixtype = header["PIXTYPE"]
        else:
            pixtype = "none"

        if pixtype.lower() == "healpix":
            if verbose:
                print(indent + "Extracting HEALPix metadata...")
            hpx_keys = ["NSIDE", "ORDERING", "NPIX", "FIRSTPIX", "LASTPIX"]
            for key in hpx_keys:
                try:
                    header_dict[key.lower()] = header[key]
                except:
                    pass
        else:
            # Assume data are non-healpix and have CTYPEn keywords
            if verbose:
                print(indent + "Extracting (RA, Dec) coordinates...")

            # FIXME: This assumes that CRPIXn references the center
            # pixel in the image which need not be the case.
            naxis = header["NAXIS"]
            ctypes = [header[f"CTYPE{i+1}"][:4] for i in range(naxis)]
            if "RA--" in ctypes and "DEC-" in ctypes:
                ra_ind = np.where([ctype == "RA--" for ctype in ctypes])[0][0]
                header_dict["s_ra"] = header[f"CTYPE{ra_ind+1}"]

                dec_ind = np.where([ctype == "DEC-" for ctype in ctypes])[0][0]
                header_dict["s_dec"] = header[f"CTYPE{dec_ind+1}"]
            elif "GLON" in ctypes and "GLAT" in ctypes:
                lon_ind = np.where([ctype == "GLON" for ctype in ctypes])[0][0]
                lat_ind = np.where([ctype == "GLAT" for ctype in ctypes])[0][0]
                sky_coord = Galactic(
                    l=header[f"CRVAL{lon_ind+1}"]*u.deg,
                    b=header[f"CRVAL{lat_ind+1}"]*u.deg
                )
                sky_coord = sky_coord.transform_to(ICRS())
                header_dict["s_ra"] = sky_coord.ra.deg
                header_dict["s_dec"] = sky_coord.dec.deg
        
        if "NAXIS1" in header:
            header_dict["s_xel1"] = header["NAXIS1"]
        if "NAXIS2" in header:
            header_dict["s_xel2"] = header["NAXIS2"]
        if "OBJECT" in header:
            header_dict["target_name"] = header["OBJECT"]
        if "TELESCOP" in header:
            header_dict["facility_name"] = header["TELESCOP"]
        header_dict["dataproduct_type"] = "image"
        header_dict["content_type"] = "fits"
    
    return header_dict

if __name__ == "__main__":
    args = get_args()
    for fp in args.file_paths:
        if args.verbose:
            print(f"Generating metadata for {fp}")
        fp = Path(fp)
        meta_dict = dict(
            namespace=args.namespace,
            name=fp.name,
            lifetime=args.lifetime,
            **get_fits_header_dict(fp, verbose=args.verbose)
        )
        out_path = Path(args.out_dir) / f"{fp.name}.meta"
        with open(out_path, "w") as f:
            json.dump(meta_dict, f)
        if args.verbose:
            print(f"Metadata file written to {out_path}", end="\n\n")
