use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Quad {
    hi: f64,
    lo: f64,
}

impl Quad {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    const SPLITTER: f64 = 134_217_729.0;

    pub fn from_f64(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    pub fn sqrt(self) -> Self {
        if self.hi == 0.0 && self.lo == 0.0 {
            return Self::ZERO;
        }
        if self.hi < 0.0 {
            return Self::from_f64(f64::NAN);
        }

        let root = Self::from_f64(self.to_f64().sqrt());
        let correction = (self - root * root) / (root * Self::from_f64(2.0));
        root + correction
    }

    fn quick_two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        let e = b - (s - a);
        Self { hi: s, lo: e }
    }

    fn two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        Self { hi: s, lo: e }
    }

    fn split(a: f64) -> (f64, f64) {
        let t = Self::SPLITTER * a;
        let hi = t - (t - a);
        let lo = a - hi;
        (hi, lo)
    }

    fn two_prod(a: f64, b: f64) -> Self {
        let p = a * b;
        let (a_hi, a_lo) = Self::split(a);
        let (b_hi, b_lo) = Self::split(b);
        let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
        Self { hi: p, lo: e }
    }
}

impl From<f64> for Quad {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl Add for Quad {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let s = Self::two_sum(self.hi, rhs.hi);
        let e = self.lo + rhs.lo + s.lo;
        Self::quick_two_sum(s.hi, e)
    }
}

impl AddAssign for Quad {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Quad {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let s = Self::two_sum(self.hi, -rhs.hi);
        let e = self.lo - rhs.lo + s.lo;
        Self::quick_two_sum(s.hi, e)
    }
}

impl SubAssign for Quad {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Quad {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let p = Self::two_prod(self.hi, rhs.hi);
        let e = self.hi * rhs.lo + self.lo * rhs.hi + p.lo;
        Self::quick_two_sum(p.hi, e)
    }
}

impl MulAssign for Quad {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Quad {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let q1 = self.hi / rhs.hi;
        let r = self - rhs * Self::from_f64(q1);
        let q2 = r.hi / rhs.hi;
        Self::from_f64(q1) + Self::from_f64(q2)
    }
}

impl DivAssign for Quad {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Quad {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}
