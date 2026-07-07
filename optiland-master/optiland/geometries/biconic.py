"""Biconic Geometry

The biconic geometry represents a surface defined by:

z = (cx * x^2 + cy * y^2) / (1 + sqrt(1 - (1 + kx) * cx^2 * x^2 - (1 + ky) * cy^2 * y^2))

where:
- cx = 1 / Rx (curvature in x)
- cy = 1 / Ry (curvature in y)
- kx is the conic constant for the x-profile
- ky is the conic constant for the y-profile

This matches the standard definition used in optical design software like Zemax.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.newton_raphson import NewtonRaphsonGeometry


class BiconicGeometry(NewtonRaphsonGeometry):
    """
    Represents a biconic geometry.

    The sag is defined by the standard biconic equation:
    z = (cx * x^2 + cy * y^2) / (1 + sqrt(1 - (1 + kx) * cx^2 * x^2 - (1 + ky) * cy^2 * y^2))

    where:
    - cx = 1 / Rx (curvature in x)
    - cy = 1 / Ry (curvature in y)
    - kx is the conic constant for the x-profile
    - ky is the conic constant for the y-profile
    """

    def __init__(
        self,
        coordinate_system: CoordinateSystem,
        radius_x: float,
        radius_y: float,
        conic_x: float = 0.0,
        conic_y: float = 0.0,
        tol: float = 1e-10,
        max_iter: int = 100,
    ):
        # Pass radius_x as the primary radius for NewtonRaphsonGeometry
        super().__init__(coordinate_system, radius_x, conic_x, tol, max_iter)

        self.Rx = be.array(radius_x)
        self.Ry = be.array(radius_y)
        self.kx = be.array(conic_x)
        self.ky = be.array(conic_y)

        self.cx = be.where(be.isinf(self.Rx) | (self.Rx == 0), 0.0, 1.0 / self.Rx)
        self.cy = be.where(be.isinf(self.Ry) | (self.Ry == 0), 0.0, 1.0 / self.Ry)

        self.is_symmetric = False  # Generally not symmetric

    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            be.ndarray or float: The sag value(s) at the given coordinates.
        """
        x = be.asarray(x)
        y = be.asarray(y)

        # Standard Biconic Sag Equation
        # z = (cx*x^2 + cy*y^2) / (1 + sqrt(1 - (1+kx)*cx^2*x^2 - (1+ky)*cy*2*y^2))

        term_x = (1.0 + self.kx) * self.cx**2 * x**2
        term_y = (1.0 + self.ky) * self.cy**2 * y**2
        
        sqrt_val = 1.0 - term_x - term_y
        # Ensure root term is non-negative
        sqrt_term = be.where(sqrt_val < 0.0, 0.0, sqrt_val)
        denom = 1.0 + be.sqrt(sqrt_term)

        # Avoid division by zero
        safe_denom = be.where(be.abs(denom) < 1e-14, 1e-14, denom)

        num = self.cx * x**2 + self.cy * y**2
        z = num / safe_denom

        return z

    def _surface_normal(self, x, y):
        """Calculate the surface normal of the geometry at the given x and y position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        x = be.asarray(x)
        y = be.asarray(y)

        # Calculate derivatives for surface normal
        # Formula: dz/dx = (x/D) * (2*cx + (z * (1+kx) * cx^2) / sqrt(Q))
        # where D = 1 + sqrt(Q), Q = 1 - (1+kx)cx^2x^2 - (1+ky)cy^2y^2

        term_x = (1.0 + self.kx) * self.cx**2 * x**2
        term_y = (1.0 + self.ky) * self.cy**2 * y**2
        Q_val = 1.0 - term_x - term_y
        
        Q = be.where(Q_val < 0.0, 0.0, Q_val)
        sqrt_Q = be.sqrt(Q)
        
        denom = 1.0 + sqrt_Q
        safe_denom = be.where(be.abs(denom) < 1e-14, 1e-14, denom)
        
        num = self.cx * x**2 + self.cy * y**2
        z = num / safe_denom
        
        safe_sqrt_Q = be.where(sqrt_Q < 1e-14, 1e-14, sqrt_Q)

        # dz/dx
        dz_dx_term2 = (z * (1.0 + self.kx) * self.cx**2) / safe_sqrt_Q
        dfdx = (x / safe_denom) * (2.0 * self.cx + dz_dx_term2)

        # dz/dy
        dz_dy_term2 = (z * (1.0 + self.ky) * self.cy**2) / safe_sqrt_Q
        dfdy = (y / safe_denom) * (2.0 * self.cy + dz_dy_term2)

        # Normal vector components (Optiland convention: (fx, fy, -1) / mag)
        mag_sq = dfdx**2 + dfdy**2 + 1.0
        mag = be.sqrt(mag_sq)
        # Avoid division by zero if mag is zero
        safe_mag = be.where(
            mag < 1e-14, 1.0, mag
        )  # if mag is ~0, normal is (0,0,-1) approx

        nx = dfdx / safe_mag
        ny = dfdy / safe_mag
        nz = -1.0 / safe_mag

        return nx, ny, nz

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radii of curvature Rx and Ry.
        Updates the curvature attributes cx and cy accordingly.
        The conic constants kx and ky remain unchanged.
        """
        self.Rx = -self.Rx
        self.Ry = -self.Ry

        # Update curvatures, handling potential division by zero if radius is zero
        self.cx = be.where(be.isinf(self.Rx) | (self.Rx == 0), 0.0, 1.0 / self.Rx)
        self.cy = be.where(be.isinf(self.Ry) | (self.Ry == 0), 0.0, 1.0 / self.Ry)

    def __str__(self) -> str:
        return "Biconic"

    def to_dict(self) -> dict:
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        # Remove base class radius and conic as they are ambiguous for biconic
        if "radius" in geometry_dict:
            del geometry_dict["radius"]
        if "conic" in geometry_dict:
            del geometry_dict["conic"]

        geometry_dict.update(
            {
                "type": self.__class__.__name__,  # Ensure correct type
                "radius_x": float(self.Rx) if hasattr(self.Rx, "item") else self.Rx,
                "radius_y": float(self.Ry) if hasattr(self.Ry, "item") else self.Ry,
                "conic_x": float(self.kx) if hasattr(self.kx, "item") else self.kx,
                "conic_y": float(self.ky) if hasattr(self.ky, "item") else self.ky,
            }
        )
        return geometry_dict

    @classmethod
    def from_dict(cls, data: dict) -> BiconicGeometry:
        """Creates a BiconicGeometry from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the biconic surface,
                containing keys like 'cs', 'radius_x', 'radius_y', 'conic_x',
                'conic_y'.

        Returns:
            BiconicGeometry: An instance of BiconicGeometry.
        """
        required_keys = {"cs", "radius_x", "radius_y"}
        if not required_keys.issubset(data):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Missing required BiconicGeometry keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])

        return cls(
            coordinate_system=cs,
            radius_x=data["radius_x"],
            radius_y=data["radius_y"],
            conic_x=data.get("conic_x", 0.0),
            conic_y=data.get("conic_y", 0.0),
            tol=data.get("tol", 1e-10),
            max_iter=data.get("max_iter", 100),
        )
