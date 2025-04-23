import math
from typing import Tuple

def vincenty_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Calculate the distance and initial bearing between two points on Earth using Vincenty's formulae.

    Args:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
        Tuple[float, float]: (distance in meters, initial bearing in degrees)
    """
    # WGS-84 ellipsoidal constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening
    b = (1 - f) * a  # semi-minor axis

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lambda1 = math.radians(lon1)
    lambda2 = math.radians(lon2)

    # Reduced latitude (latitude on the auxiliary sphere)
    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))

    # Difference in longitude
    L = lambda2 - lambda1
    lambda_old = L

    # Initialize variables
    sin_sigma = cos_sigma = sigma = sin_alpha = cos_sq_alpha = cos2_sigma_m = 0

    # Iterate until convergence
    for i in range(100):
        sin_sigma = math.sqrt(
            (math.cos(U2) * math.sin(L)) ** 2 +
            (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(L)) ** 2
        )

        if sin_sigma == 0:
            return 0.0, 0.0  # Coincident points

        cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * math.cos(L)
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = math.cos(U1) * math.cos(U2) * math.sin(L) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2

        if cos_sq_alpha != 0:
            cos2_sigma_m = cos_sigma - 2 * math.sin(U1) * math.sin(U2) / cos_sq_alpha
        else:
            cos2_sigma_m = 0  # Equatorial line

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

        lambda_new = L + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (
                cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)
        )
        )

        if abs(lambda_new - lambda_old) < 1e-12:
            break  # Convergence achieved
        lambda_old = lambda_new
    else:
        raise ValueError("Vincenty formula failed to converge")

    # Calculate distance
    u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    delta_sigma = B * sin_sigma * (
            cos2_sigma_m + B / 4 * (
            cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) -
            B / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma ** 2) *
            (-3 + 4 * cos2_sigma_m ** 2)
    )
    )

    distance = b * A * (sigma - delta_sigma)

    # Calculate initial bearing
    initial_bearing = math.atan2(
        math.cos(U2) * math.sin(L),
        math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(L)
    )
    initial_bearing = math.degrees(initial_bearing)
    initial_bearing = (initial_bearing + 360) % 360  # Normalize to 0-360 degrees

    return distance, initial_bearing


