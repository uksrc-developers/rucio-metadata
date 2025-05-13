"""
Generate a companion metadata file for uploading data via Rucio.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from astropy.coordinates import Galactic, ICRS
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import astropy_healpix.healpy as healpy
from casacore.tables import table
import h5py
import json
import numpy as np
from pathlib import Path
from pprint import pprint

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
    parser.add_argument(
        "--calib_level",
        type=int,
        default=2,
        help="Calibration level. Defaults to 2."
    )
    parser.add_argument(
        "--obs_collection",
        type=str,
        default="teal-testing",
        help="Observation collection. Defaults to 'teal-testing'."
    )
    parser.add_argument(
        "--dataproduct_type",
        type=str,
        help="Type of data product (e.g. image, catalog). If not provided, "
             "inferred from file extension."
    )
    parser.add_argument(
        "--access_format",
        type=str,
        help="MIME-type-compatible access format string. Defaults to "
             "'application/{file extension}'."
    )
    return parser.parse_args()

# Function for a .fits file
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
            hpx_keys = ["nside", "ordering", "npix", "firstpix", "lastpix"]
            for key in hpx_keys:
                try:
                    header_dict[key] = header[key.upper()]
                except:
                    pass
            if np.all([key in header_dict for key in hpx_keys]):
                # Pull the (RA, Dec) of the central pixel in the HEALPix map
                lon, lat = healpy.pix2ang(
                    header_dict["nside"],
                    (header_dict["firstpix"] + header_dict["lastpix"])//2,
                    nest=header_dict["ordering"].lower()=="nest",
                    lonlat=True
                )
                header_dict["s_ra"] = lon
                header_dict["s_dec"] = lat
        else:
            if verbose:
                print(indent + "Extracting (RA, Dec) coordinates...")            
            try:
                wcs = WCS(header)
                coord = wcs.array_index_to_world(
                    wcs.array_shape[0]//2, wcs.array_shape[1]//2
                )
                header_dict["s_ra"] = coord.icrs.ra.deg
                header_dict["s_dec"] = coord.icrs.dec.deg
            except:
                # FIXME: This assumes that CRPIXn references the center
                # pixel in the image which need not be the case.
                naxis = header["NAXIS"]
                ctypes = []
                for i in range(naxis):
                    ctype_key = f"CTYPE{i+1}"
                    if ctype_key in header:
                        ctypes.append(header[ctype_key][:4])
                    else:
                        ctypes.append("")
                if "RA--" in ctypes:
                    ra_str = "RA--"
                elif "RA" in ctypes:
                    ra_str = "RA"
                if "DEC-" in ctypes:
                    dec_str = "DEC-"
                elif "DEC" in ctypes:
                    dec_str = "DEC"
                if ra_str in ctypes and dec_str in ctypes:
                    ra_ind = np.where([ctype == ra_str for ctype in ctypes])[0][0]
                    header_dict["s_ra"] = header[f"CRVAL{ra_ind+1}"]

                    dec_ind = np.where([ctype == dec_str for ctype in ctypes])[0][0]
                    header_dict["s_dec"] = header[f"CRVAL{dec_ind+1}"]
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
    
    return header_dict

# Function for a .hdf5 file
def get_hdf5_metadata(fpath, verbose=False, indent="\t") -> dict:
    """
    Extract basic metadata from an HDF5 file.

    Parameters
    ----------
    fpath : Path or str
        Path to the HDF5 file.
    verbose : bool
        Verbose output.
    indent : str
        Indentation for verbose output.

    Returns
    -------
    header_dict : dict
        Dictionary of extracted metadata.
    """
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist.")

    header_dict = {}
    with h5py.File(fpath, "r") as hdf:
        # Attempt to collect top-level attributes
        for key, val in hdf.attrs.items():
            try:
                header_dict[key] = val.item() if hasattr(val, "item") else val
            except Exception:
                pass

        # Optional: count top-level datasets/groups
        header_dict["num_groups"] = len([k for k in hdf.keys()])

    if verbose:
        print(
            indent
            + f"Extracted {len(header_dict)} metadata entries from HDF5."
        )
    
    return header_dict

# Function for a .ms file
def get_ms_metadata(fpath, verbose=False, indent="\t") -> dict:
    """
    Extract basic metadata from a Measurement Set (MS).

    Parameters
    ----------
    fpath : Path or str
        Path to the MS directory.
    verbose : bool
        Verbose output.
    indent : str
        Indentation for verbose output.

    Returns
    -------
    header_dict : dict
        Dictionary of extracted metadata.
    """
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist.")

    header_dict = {}
    try:
        main = table(str(fpath))
        try:
            header_dict["target_name"] = str(main.getcol("FIELD_ID")[0])
        except:
            pass
        try:
            header_dict["facility_name"] = str(main.getcol("ANTENNA1")[0])
        except:
            pass
        main.close()

        if verbose:
            print(indent + f"Extracted MS metadata.")
    except Exception as e:
        if verbose:
            print(indent + f"Error reading MS: {e}")

    return header_dict


if __name__ == "__main__":
    args = get_args()
    for fp in args.file_paths:
        if args.verbose:
            print(f"Generating metadata for {fp}")
        fp = Path(fp)
        
        # Determining content_type and acess format
        suffix = fp.suffix.lower().lstrip(".")  # e.g. 'fits', 'hdf5', 'ms'
        content_type = suffix
        access_format = args.access_format or f"application/{suffix}"

        # Infer dataproduct_type from extension if not provided
        if args.dataproduct_type is not None:
            dataproduct_type = args.dataproduct_type
        else:
            ext_map = {
                "fits": "fits",
                "uvfits": "fits",
                "hdf5": "hdf5",
                "h5": "hdf5",
                "ms": "measurement set"
            }
            dataproduct_type = ext_map.get(suffix, "unknown")

        # Determine the correct header function to obtain the fields
        if suffix in ("fits", "fit", "fts", "uvfits"):
            header_info = get_fits_header_dict(fp, verbose=args.verbose)
        elif suffix in ("hdf5", "h5"):
            header_info = get_hdf5_metadata(fp, verbose=args.verbose)
        elif suffix == "ms":
            header_info = get_ms_metadata(fp, verbose=args.verbose)
        else:
            header_info = {}

        # URLs from obs_id to move around in Rucio
        obs_id = f"{args.namespace}:{fp.name}"
        obs_publisher_did = f"ivo://test.skao/~?{obs_id}"
        access_url = f"https://datalink.ivoa.srcnet.skao.int/rucio/links?id={obs_id}"
        
        # Forming the dictionary for the .json file
        meta_dict = dict(
            namespace=args.namespace,
            name=fp.name,
            lifetime=args.lifetime,
            calib_level=args.calib_level,
            obs_collection=args.obs_collection,
            obs_id=obs_id,
            obs_publisher_did=obs_publisher_did,
            access_url=access_url,
            access_format=access_format,
            content_type=content_type,
            dataproduct_type=dataproduct_type,
            **header_info
        )
        out_path = Path(args.out_dir) / f"{fp.name}.meta"
        with open(out_path, "w") as f:
            json.dump(meta_dict, f, indent = 4)
        if args.verbose:
            print(f"Metadata file written to {out_path}", end="\n\n")
            pprint(meta_dict)
            print()

