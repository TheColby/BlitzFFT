use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Quad(pub f128);

impl Quad {
    pub const ZERO: Self = Self(0.0);
    pub const ONE: Self = Self(1.0);
    pub const PI: Self = Self(3.1415926535897932384626433832795028842);
    pub const TWO_PI: Self = Self(6.2831853071795864769252867665590057684);
    pub const HALF_PI: Self = Self(1.5707963267948966192313216916397514421);
    const SERIES_EPSILON: Self = Self(1e-34);

    pub fn from_f64(value: f64) -> Self {
        Self(value as f128)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64
    }

    pub fn sqrt(self) -> Self {
        if self.0 == 0.0 {
            return Self::ZERO;
        }
        if self.0 < 0.0 {
            return Self(f128::NAN);
        }

        let mut x = Self::from_f64(self.to_f64().sqrt());
        if x.0 == 0.0 {
            x = Self::ONE;
        }

        let half = Self::from(0.5);
        for _ in 0..12 {
            x = half * (x + self / x);
        }
        x
    }

    pub fn abs(self) -> Self {
        if self.0 < 0.0 { -self } else { self }
    }

    pub fn sin(self) -> Self {
        let (sin, _) = self.sin_cos();
        sin
    }

    pub fn cos(self) -> Self {
        let (_, cos) = self.sin_cos();
        cos
    }

    pub fn sin_cos(self) -> (Self, Self) {
        let mut x = self.wrap_angle();

        if x > Self::HALF_PI {
            let reflected = Self::PI - x;
            let (sin, cos) = reflected.sin_cos_taylor();
            return (sin, -cos);
        }

        if x < -Self::HALF_PI {
            x = -Self::PI - x;
            let (sin, cos) = x.sin_cos_taylor();
            return (-sin, -cos);
        }

        x.sin_cos_taylor()
    }

    fn wrap_angle(self) -> Self {
        let mut x = self;
        while x > Self::PI {
            x -= Self::TWO_PI;
        }
        while x < -Self::PI {
            x += Self::TWO_PI;
        }
        x
    }

    fn sin_cos_taylor(self) -> (Self, Self) {
        let x2 = self * self;

        let mut sin_term = self;
        let mut sin_sum = sin_term;
        let mut k = 1u32;
        loop {
            let denom = Self::from((2 * k * (2 * k + 1)) as f64);
            sin_term *= -x2 / denom;
            sin_sum += sin_term;
            if sin_term.abs() < Self::SERIES_EPSILON || k >= 48 {
                break;
            }
            k += 1;
        }

        let mut cos_term = Self::ONE;
        let mut cos_sum = cos_term;
        let mut k = 1u32;
        loop {
            let denom = Self::from(((2 * k - 1) * (2 * k)) as f64);
            cos_term *= -x2 / denom;
            cos_sum += cos_term;
            if cos_term.abs() < Self::SERIES_EPSILON || k >= 48 {
                break;
            }
            k += 1;
        }

        (sin_sum, cos_sum)
    }
}

impl From<f64> for Quad {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<f32> for Quad {
    fn from(value: f32) -> Self {
        Self(value as f128)
    }
}

impl Add for Quad {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Quad {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Quad {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Quad {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for Quad {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for Quad {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for Quad {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign for Quad {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Neg for Quad {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}
