export const COMPLEX_WGSL = /* wgsl */ `
const PI: f32 = 3.1415926535897932384626433832795;

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b;
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

fn cis(angle: f32) -> vec2<f32> {
  return vec2<f32>(cos(angle), sin(angle));
}
`;

