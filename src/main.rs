use rand::Rng;
use rayon::iter::Fold;
use std::ops;
use std::f32::consts;
use std::fs;
use std::iter;
use std::ops::Rem;
use rayon::prelude::*;

const MAX_T: f32 = 20000.0;

fn main() {
    let r = (std::f32::consts::PI / 4.0).cos();
    // Image
    let aspect_ratio = 3.0 / 2.0;
    let image_width = 1200;
    let image_height: i32 = (image_width as f32 / aspect_ratio) as i32;
    let samples_per_pixel = 500;
    let max_depth = 50;

    // World
    let material_ground = Lambertian {
        albedo: Colour {
            x: 0.8,
            y: 0.8,
            z: 0.0,
        },
    };
    let material_centre = Lambertian {
        albedo: Colour {
            x: 0.1,
            y: 0.2,
            z: 0.5,
        },
    };
    // let material_left = Metal{albedo:Colour{x: 0.8, y:0.8, z:0.8}, fuzz: 0.3};
    let material_left = Dielectric {
        index_of_refraction: 1.5,
    };
    let material_right = Metal {
        albedo: Colour {
            x: 0.8,
            y: 0.6,
            z: 0.2,
        },
        fuzz: 1.0,
    };

    // let material_left = Lambertian{albedo:Colour{x: 0.0, y:0.0, z:1.0}};
    // let material_right = Lambertian{albedo:Colour{x: 1.0, y:0.0, z:0.0}};

    // let mut world_list: Vec<Box<dyn HittableObject>> = Vec::new();
    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 {
    //         x: 0.0,
    //         y: -100.5,
    //         z: -1.0,
    //     },
    //     radius: 100.0,
    //     material: Material::Lambertian(material_ground),
    // }));
    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 {
    //         x: 0.0,
    //         y: 0.0,
    //         z: -1.0,
    //     },
    //     radius: 0.5,
    //     material: Material::Lambertian(material_centre),
    // }));
    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 {
    //         x: -1.0,
    //         y: 0.0,
    //         z: -1.0,
    //     },
    //     radius: 0.5,
    //     material: Material::Dielectric(material_left),
    // }));
    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 {
    //         x: -1.0,
    //         y: 0.0,
    //         z: -1.0,
    //     },
    //     radius: -0.4,
    //     material: Material::Dielectric(material_left),
    // }));
    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 {
    //         x: 1.0,
    //         y: 0.0,
    //         z: -1.0,
    //     },
    //     radius: 0.5,
    //     material: Material::Metal(material_right),
    // }));

    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 { x: -r, y: 0.0, z: -1.0 },
    //     radius: r,
    //     material: Material::Lambertian(material_left),
    // }));
    // world_list.push(Box::new(Sphere {
    //     centre: Vec3 { x: r, y: 0.0, z: -1.0 },
    //     radius: r,
    //     material: Material::Lambertian(material_right),
    // }));

    let build_spheres = many_spheres();
    let world = HittableList {
        objects: &build_spheres,
    };

    // let world = HittableList {
    //     objects: &world_list,
    // };

    let look_from = Point3 {
        x: 13.0,
        y: 2.0,
        z: 3.0,
    };
    let look_at = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let v_up = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let camera = build_camera(
        look_from,
        look_at,
        v_up,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );

    // Render
    println!("P3");
    println!("{}", image_width.to_string());
    println!("{}", image_height.to_string());
    println!("255");

    // let mut rng  = StdRng::from_entropy();

    // See https://raytracing.github.io/images/fig-1.03-cam-geom.jpg
    for j in (0..image_height).rev() {
        for i in 0..image_width {
            // let mut pixel_colour = Colour {
            //     x: 0.0,
            //     y: 0.0,
            //     z: 0.0,
            // };

            let aggregated_pixel = (0..samples_per_pixel).into_par_iter().map(|sample| {
                let u = (i as f32 + fake_random(sample, true)) / (image_width as f32 - 1.0); // why minus one?
                let v = (j as f32 + fake_random(sample, false)) / (image_height as f32 - 1.0); // why minus one?
                let ray = camera.get_ray(u, v);
                return ray_colour(
                    ray,
                    &world,
                    max_depth,
                    fake_random(sample, false) * 100.0,
                )
            }
            ).reduce(|| Colour{x:0.0,y:0.0,z:0.0},
                |accum, color| accum.add(&color));


            // for sample in 0..samples_per_pixel {
            //     let u = (i as f32 + fake_random(sample, true)) / (image_width as f32 - 1.0); // why minus one?
            //     let v = (j as f32 + fake_random(sample, false)) / (image_height as f32 - 1.0); // why minus one?
            //     let ray = camera.get_ray(u, v);
            //     pixel_colour = pixel_colour.add(&ray_colour(
            //         ray,
            //         &world,
            //         max_depth,
            //         fake_random(sample, false) * 100.0,
            //     ));
            // }
            write_pixel(aggregated_pixel, samples_per_pixel);
        }
    }
}

fn many_spheres() -> Vec<Box<dyn HittableObject>> {
    let material_ground = Lambertian {
        albedo: Colour {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        },
    };
    let material_lambertian = Lambertian {
        albedo: Colour {
            x: 0.4,
            y: 0.2,
            z: 0.1,
        },
    };
    // let material_left = Metal{albedo:Colour{x: 0.8, y:0.8, z:0.8}, fuzz: 0.3};
    let material_glass = Dielectric {
        index_of_refraction: 1.5,
    };

    // let material_left = Lambertian{albedo:Colour{x: 0.0, y:0.0, z:1.0}};
    // let material_right = Lambertian{albedo:Colour{x: 1.0, y:0.0, z:0.0}};

    let mut world_list: Vec<Box<dyn HittableObject>> = Vec::new();
    // Add "ground" sphere.
    world_list.push(Box::new(Sphere {
        centre: Vec3 {
            x: 0.0,
            y: -1000.0,
            z: 0.0,
        },
        radius: 1000.0,
        material: Material::Lambertian(material_ground),
    }));

    let world_size = 11;

    for a in -world_size..world_size {
        for b in -world_size..world_size {
            let noise = random_unit();
            let centre = Vec3 {
                x: random_unit() * 0.9 + a as f32,
                y: 0.2,
                z: random_unit() * 0.9 + b as f32,
            };

            let threshold_point = Point3 {
                x: 4.0,
                y: 0.2,
                z: 0.0,
            };
            // Don't render if we're too near?
            if centre.subtract(&threshold_point).length() > 0.9 {
                if noise < 0.8 {
                    // Diffuse material
                    world_list.push(Box::new(Sphere {
                        centre: centre,
                        radius: 0.2,
                        material: Material::Lambertian(random_lambertian()),
                    }));
                } else if noise < 0.95 {
                    // Metal
                    world_list.push(Box::new(Sphere {
                        centre: centre,
                        radius: 0.2,
                        material: Material::Metal(random_metal()),
                    }));
                } else {
                    //glass
                    world_list.push(Box::new(Sphere {
                        centre: centre,
                        radius: 0.2,
                        material: Material::Dielectric(material_glass),
                    }));
                }
            }
        }
    }

    world_list.push(Box::new(Sphere {
        centre: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        radius: 1.0,
        material: Material::Dielectric(material_glass),
    }));
    world_list.push(Box::new(Sphere {
        centre: Vec3 {
            x: -4.0,
            y: 1.0,
            z: 0.0,
        },
        radius: 1.0,
        material: Material::Lambertian(material_lambertian),
    }));
    let material_metal = Metal {
        albedo: Colour {
            x: 0.7,
            y: 0.6,
            z: 0.5,
        },
        fuzz: 0.0,
    };
    world_list.push(Box::new(Sphere {
        centre: Vec3 { x: 4.0, y: 1.0, z: 0.0 },
        radius: 1.0,
        material: Material::Metal(material_metal),
    }));
    return world_list;
}

fn random_lambertian() -> Lambertian {
    // Sample code does:
    // auto albedo = color::random() * color::random();
    // which is a pointwise multipleication of random colours.
    return Lambertian {
        albedo: Colour {
            x: random_unit() * random_unit(),
            y: random_unit() * random_unit(),
            z: random_unit() * random_unit(),
        },
    };
}

fn random_metal() -> Metal {
    let mut rng = rand::thread_rng();

    let fuzz = rng.gen_range(0.0..0.5);
    return Metal {
        albedo: Colour {
            x: rng.gen_range(0.5..1.0),
            y: rng.gen_range(0.5..1.0),
            z: rng.gen_range(0.5..1.0),
        },
        fuzz: fuzz,
    };
}

// fn ray_colour(ray: Ray) -> Colour {
//     // println!("ray.direction.y is {}",ray.direction.y);
//     let unit_direction = ray.direction.unit_vector();
//     let t = 0.5 * (unit_direction.y + 1.0);
//     let white = Colour{x: 1.0, y: 1.0, z:1.0};
//     let colour2 = Colour{x: 0.1, y: 0.3, z:1.0};
//     // let colour2 = Colour{x: 0.5, y: 0.7, z:1.0};

//     // red =  x = (1-t)*1 + t*0.5 = 1 - t*(0.5)
//     // green =  y = (1-t)*1 + t*0.7 = 1 - t*(0.3)
//     // blue =  z = (1-t)*1 + t*1.0 = 1
//     // So the same amount of blue, and less and less red and green as t increases
//     return white.multiply(1.0 - t).add(&colour2.multiply(t));
// }

// fn write_data() -> String {

//     let mut header: Vec<String> = vec!["P3".to_string(),
//     "255".to_string(), // width
//     "200".to_string(), // height
//     "255".to_string(), //max value for each colour
//     ];

//     header.append(&mut write_lines(200));
//     return header.join("\n")
// }

// fn write_lines(height: usize) -> Vec<String> {
//     return iter::repeat(write_line(255, 128)).take(height).collect::<Vec<String>>();

// }

// fn write_line(width: usize, green: usize) -> String {
//     return (0..width).map(|x| write_pixel(x, green, 64)).collect::<Vec<String>>().join(" ");
//     // return iter::repeat(write_pixel(125, green, 64)).take(length).collect::<Vec<String>>().join(" ");
// }

// original, without antialiasing
// fn write_pixel(colour: Colour) {
//     let out = [
//         colour.x_for_printing(),
//         colour.y_for_printing(),
//         colour.z_for_printing(),
//     ]
//     .join(" ");
//     println!("{}", out);
// }

fn write_pixel(colour: Colour, samples_per_pixel: i32) {
    let scale = 1.0 / (samples_per_pixel as f32);
    // Sqrt for gamma correction.
    let r = (colour.x * scale).sqrt();
    let g = (colour.y * scale).sqrt();
    let b = (colour.z * scale).sqrt();

    let out = [
        safe_colour_print(r),
        safe_colour_print(g),
        safe_colour_print(b),
    ]
    .join(" ");
    println!("{}", out);
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
fn cross(u: Vec3, v: Vec3) -> Vec3 {
    return Vec3 {
        x: (u.y * v.z) - u.z * v.y,
        y: (u.z * v.x) - u.x * v.z,
        z: (u.x * v.y) - u.y * v.x,
    };
}

// You can form a vector quadratic equation using the extension of the Ray (which is itself a function of time, t),
// This gives us an equation for whether there is any T such that the ray strikes the sphere. This quadratic has
// either 0, 1 or 2 solutions, corresponding to mising, clipping, or going through the sphere
//
// The a, b, c components of the quadratic are derived here:
// https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/creatingourfirstraytracedimage
// (The half_b optimisation is described in section 6.2.)
//
// Returns the time t at which the ray first hit the sphere.
fn hit_sphere(centre: Vec3, radius: f32, ray: &Ray) -> Option<f32> {
    let origin_centre = ray.origin.subtract(&centre);
    let a = dot(ray.direction, ray.direction);
    let half_b = dot(origin_centre, ray.direction);
    // let b = 2.0 * dot(origin_centre, ray.direction);
    let c = dot(origin_centre, origin_centre) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    // return discriminant > 0.0;
    if discriminant < 0.0 {
        return None;
    } else {
        // Returning first hit, which is lowest t, so -b - sqrt(disc)/2a
        return Some((-half_b - discriminant.sqrt()) / a);
    }
}

/// Colour a ray depending on if it hits a sphere at the centre of our viewport.
/// If not hit (i.e. time_hit is None), display background gradient as in commented
// out ray_colour.
// If it does hit, colour depending on the direction of the normal at the point
// of impact.
// The unit normal is given by the point of contact, P, minus the sphere centre.
// (If you're standing at P on earth, the direction to the centre of the earth is
// C - P, so the opposite direction is P-C).
// The chose colour is to take the unit normal and use it's parameters as colours.
// TODO the seed is only needed because I'm using my pseudorandom function. Can be deleted when
// I use real random.
fn ray_colour(ray: Ray, world: &HittableList, depth: i32, seed: f32) -> Colour {
    if depth <= 0 {
        return Colour {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

    let maybe_hit_record = world.hit(&ray, 0.001, MAX_T);
    match maybe_hit_record {
        Some(hit_record) => {
            let maybe_reflection = hit_record.material.scatter(ray, &hit_record);
            match maybe_reflection {
                //todo rename colour to attenutation when i understand what that is.
                Some((reflected_ray, surface_colour)) => {
                    let reflection_colour = ray_colour(reflected_ray, world, depth - 1, seed);
                    return Colour {
                        x: surface_colour.x * reflection_colour.x,
                        y: surface_colour.y * reflection_colour.y,
                        z: surface_colour.z * reflection_colour.z,
                    };
                }

                None => {
                    return Colour {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    }
                }
            }

            // // This is where the reflection bounces to, featuring a random variation.
            // let target = hit_record.p
            //     .add(&hit_record.normal)
            //     .add(&random_unit_vector());

            // let reflected_ray = Ray {
            //     origin: hit_record.p,
            //     //
            //     direction: target.subtract(&hit_record.p),
            // };
            // let colour_at_target = ray_colour(reflected_ray, world, depth - 1, seed + 1.0);

            // return colour_at_target.multiply(0.5);
        }

        None => {
            //see other ray colour function
            let unit_direction = ray.direction.unit_vector();
            let t = 0.5 * (unit_direction.y + 1.0);
            let white = Colour {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            };
            let colour2 = Colour {
                x: 0.5,
                y: 0.7,
                z: 1.0,
            };

            return white.multiply(1.0 - t).add(&colour2.multiply(t));
        }
    }
}

type Colour = Vec3;
type Point3 = Vec3;

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

// impl  ops::Add<Vec3> for Vec3 {
//     fn add(self, to_add: Vec3) -> Vec3 {
//         return self.add(to_add);
//     }

//     type Output = Vec3;
    
// }

impl Vec3 {
    fn add(&self, other: &Vec3) -> Vec3 {
        return Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }

    fn subtract(&self, other: &Vec3) -> Vec3 {
        return Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        };
    }

    fn multiply(&self, scalar: f32) -> Vec3 {
        return Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        };
    }

    fn divide(&self, scalar: f32) -> Vec3 {
        return self.multiply(1.0 / scalar);
    }

    fn length_squared(&self) -> f32 {
        return self.x.powi(2) + self.y.powi(2) + self.z.powi(2);
    }

    fn length(&self) -> f32 {
        return self.length_squared().sqrt();
    }

    fn unit_vector(&self) -> Vec3 {
        return self.divide(self.length());
    }

    //for printing colours only
    fn x_for_printing(&self) -> String {
        // return ((255.999 * self.x) as i32).to_string();
        return self.safe_colour_print(self.x);
    }
    fn y_for_printing(&self) -> String {
        // return ((255.999 * self.y) as i32).to_string();
        return self.safe_colour_print(self.y);
    }
    fn z_for_printing(&self) -> String {
        // return ((255.999 * self.z) as i32).to_string();
        return self.safe_colour_print(self.z);
    }

    // Check that printing is outputing valid integer.
    // Should be "Static" equivalent, but I'm on a plane and can't look it up.
    fn safe_colour_print(&self, coord: f32) -> String {
        let as_i32 = (255.999 * coord) as i32;
        assert!(as_i32 >= 0, "failed to print because as_i32 is {}", as_i32);
        assert!(as_i32 < 256, "failed to print because as_i32 is {}", as_i32);
        return as_i32.to_string();
    }

    fn near_zero(&self) -> bool {
        let threshold = 1e-8;
        return self.x < threshold && self.y < threshold && self.z < threshold;
    }
}

//todo have this or static method above but not both.
fn safe_colour_print(coord: f32) -> String {
    let as_i32 = (255.999 * coord) as i32;
    assert!(as_i32 >= 0, "failed to print because as_i32 is {}", as_i32);
    assert!(as_i32 < 256, "failed to print because as_i32 is {}", as_i32);
    return as_i32.to_string();
}

struct Ray {
    origin: Point3,
    direction: Point3,
}

impl Ray {
    fn at(&self, t: f32) -> Point3 {
        return self.origin.add(&self.direction.multiply(t));
    }
}

fn hit_record_with_norml<'a>(
    p: Vec3,
    t: f32,
    material: &Material,
    ray: &Ray,
    outward_normal: Vec3,
) -> HitRecord {
    let is_front_face = dot(ray.direction, outward_normal) < 0.0;
    let normal = if is_front_face {
        outward_normal
    } else {
        outward_normal.multiply(-1.0)
    };
    return HitRecord {
        p,
        normal,
        t,
        is_front_face,
        material: (*material).clone(),
    };
}

struct HitRecord {
    // P is the point of hitting
    p: Point3,
    normal: Vec3,
    t: f32,
    is_front_face: bool,
    material: Material,
}

impl HitRecord {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        // If the ray direciton and outward normal are in opposite direction (dot
        // product < 0), then the ray is hitting the outside.
        self.is_front_face = dot(ray.direction, outward_normal) < 0.0;
        self.normal = if self.is_front_face {
            outward_normal
        } else {
            outward_normal.multiply(-1.0)
        };
    }
}

#[derive()]
struct HittableList<'a> {
    objects: &'a Vec<Box<dyn HittableObject>>,
}

impl HittableObject for HittableList<'_> {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut temp_record = None;
        let mut hit_anything = false;
        let mut closest_so_far = t_max;

        for scene_object in self.objects {
            let this_hit_record = scene_object.hit(ray, t_min, closest_so_far);
            match this_hit_record {
                Some(hit_record) => {
                    closest_so_far = hit_record.t;
                    hit_anything = true;
                    temp_record = Some(hit_record);
                }
                None => (),
            }
        }
        return temp_record;
    }
}

trait HittableObject : Sync {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

struct Sphere {
    centre: Point3,
    radius: f32,
    material: Material,
}

impl HittableObject for Sphere {
    /// See code on hit_sphere for logic and and quadratic optimisation
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let origin_centre = ray.origin.subtract(&self.centre);
        let a = dot(ray.direction, ray.direction);
        let half_b = dot(origin_centre, ray.direction);
        // let b = 2.0 * dot(origin_centre, ray.direction);
        let c = dot(origin_centre, origin_centre) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        // return discriminant > 0.0;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();

        let mut root = (-half_b - sqrt_discriminant) / a;
        if root < t_min || root > t_max {
            // First root invalid, try again
            root = (-half_b + sqrt_discriminant) / a;
            if root < t_min || root > t_max {
                // Other solution is a miss too
                return None;
            }
        }
        let p = ray.at(root);
        let outward_normal = p.subtract(&self.centre).divide(self.radius);
        return Some(hit_record_with_norml(
            p,
            root,
            &self.material,
            ray,
            outward_normal,
        ));
    }
}

struct Camera {
    aspect_ratio: f32,
    viewport_height: f32,
    viewport_width: f32,
    focal_length: f32,

    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,

    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
}

impl Camera {
    fn get_ray(&self, s: f32, t: f32) -> Ray {
        let rd = random_in_unit_disk().multiply(self.lens_radius);
        let offset = self.u.multiply(rd.x).add(&&self.v.multiply(rd.y));

        let ray_direction = self
            .lower_left_corner
            .add(&self.horizontal.multiply(s as f32))
            .add(&self.vertical.multiply(t as f32))
            .subtract(&self.origin)
            .subtract(&offset);

        let ray = Ray {
            origin: self.origin.add(&offset),
            direction: ray_direction,
        };
        return ray;
    }
}

// look_from is the position of the camera.
// look_at is where the camera is pointing to (note: it is a point, rather than a direction)
// v_up is an "up" vector for the camera. look_from and look_at only specify
// the camera within a rotation (think of tilting your head while fixing your gaze)
fn build_camera(
    look_from: Point3,
    look_at: Point3,
    v_up: Vec3,
    vertical_field_of_view: f32,
    aspect_ratio: f32,
    aperture: f32,
    focus_dist: f32,
) -> Camera {
    let theta = degrees_to_radians(vertical_field_of_view);
    let h = (theta / 2.0).tan();
    let viewport_height = 2.0 * h;
    let viewport_width = aspect_ratio * viewport_height;

    // We're going to build a basis for the vector space.
    // w is s.t. -w is the camera angle.
    let w = look_from.subtract(&look_at).unit_vector();
    let u = cross(v_up, w);
    let v = cross(w, u);

    // let viewport_height = 2.0;
    // let aspect_ratio = 16.0 / 9.0;
    // let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = look_from;
    let horizontal = u.multiply(viewport_width).multiply(focus_dist);
    let vertical = v.multiply(viewport_height).multiply(focus_dist);

    let half_plane = horizontal.multiply(0.5).add(&vertical.multiply(0.5));
    let lower_left_corner = origin
        .subtract(&half_plane)
        .subtract(&&w.multiply(focus_dist));

    let lens_radius = aperture / 2.0;

    return Camera {
        aspect_ratio: 16.0 / 9.0,
        viewport_height,
        viewport_width,
        focal_length,

        origin,
        horizontal,
        vertical,
        lower_left_corner,

        u,
        v,
        w,
        lens_radius,
    };
}

fn degrees_to_radians(degrees: f32) -> f32 {
    return degrees * std::f32::consts::PI / 180.0;
}

fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        return min;
    }
    if x > max {
        return max;
    }
    return x;
}

// This is meant to loosely be a random function, because I'm on a plane
// and a) can't look up how you actually do random in
// rust  and b) can't install the rand package.
fn fake_random(sample: i32, is_u: bool) -> f32 {
    let seed = (sample + if is_u { 1 } else { 2 }) as f32;
    return fake_random_unit(seed);
}

// fn random_unit() -> f32 {
//     let mut rn
// }

// Returns a pseudorandom number in [0, 1) using pi as "randomness"
// source.
fn fake_random_unit(seed: f32) -> f32 {
    let times_pi = seed * std::f32::consts::PI;

    let integer_part = times_pi.floor() as i32;

    let randomly_negative = (integer_part % 2) == 0;

    let fraction_part = times_pi - (integer_part as f32);

    assert!(fraction_part < 1.0);
    assert!(fraction_part >= 0.0);

    return fraction_part * if randomly_negative { -1.0 } else { 1.0 };
}

fn random_unit() -> f32 {
    let mut rng = rand::thread_rng();
    return rng.gen_range(0.0..1.0);
}

fn random_vec3() -> Vec3 {
    let mut rng = rand::thread_rng();

    return Vec3 {
        x: rng.gen_range(0.0..1.0),
        y: rng.gen_range(0.0..1.0),
        z: rng.gen_range(0.0..1.0),
    };
}

fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = random_vec3();
        let q = Vec3 {
            x: p.x,
            y: p.y,
            z: 0.0,
        };
        if q.length_squared() >= 1.0 {
            continue;
        }
        return q;
    }
}

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = random_vec3();
        if p.length_squared() >= 1.0 {
            continue;
        }
        return p;
    }
}

fn random_unit_vector() -> Vec3 {
    return random_in_unit_sphere().unit_vector();
}

#[derive(Clone, Copy)]
enum Material {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
}

impl Material {
    fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        match *self {
            Material::Lambertian(lamberian) => lamberian.scatter(ray_in, hit_record),
            Material::Metal(metal) => metal.scatter(ray_in, hit_record),
            Material::Dielectric(dielectric) => dielectric.scatter(ray_in, hit_record),
        }
    }
}

// trait Material: Clone  {
//     fn scatter(&self,ray_in : Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)>;
// }

#[derive(Clone, Copy)]
struct Lambertian {
    albedo: Colour,
}

// impl Material for Lambertian {
impl Lambertian {
    fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        let mut scatter_direction = hit_record.normal.add(&random_unit_vector());
        if scatter_direction.near_zero() {
            scatter_direction = hit_record.normal;
        }

        let scattered_ray = Ray {
            origin: hit_record.p,
            direction: scatter_direction,
        };

        return Some((scattered_ray, self.albedo));
    }
}

// I didn't really follow the whole derivation of this.
// See section 10.2
fn refract(uv: Vec3, normal: Vec3, eta_over_eta_prime: f32) -> Vec3 {
    let cos_theta = f32::min(dot(uv.multiply(-1.0), normal), 1.0);
    let cos_theta_n = normal.multiply(cos_theta);
    let r_out_perp = cos_theta_n.add(&uv).multiply(eta_over_eta_prime);
    let multiplier = (1.0 - r_out_perp.length_squared()).sqrt();
    let r_out_parallel = normal.multiply(-1.0 * multiplier);
    return r_out_perp.add(&r_out_parallel);
}

#[derive(Clone, Copy)]
struct Metal {
    albedo: Colour,
    fuzz: f32,
}

// impl Material for Metal {
impl Metal {
    fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        let reflected = reflect(ray_in.direction, hit_record.normal);
        let noise = &random_in_unit_sphere().multiply(self.fuzz);
        let scattered_ray = Ray {
            origin: hit_record.p,
            direction: reflected.add(noise),
        };

        // Hmm, how could this happen? I think maybe if the ray hit the inside
        // of the surface.
        if dot(scattered_ray.direction, hit_record.normal) < 0.0 {
            return None;
        }
        return Some((scattered_ray, self.albedo));
    }
}

#[derive(Clone, Copy)]
struct Dielectric {
    index_of_refraction: f32,
}
impl Dielectric {
    fn scatter(&self, ray_in: Ray, hit_record: &HitRecord) -> Option<(Ray, Colour)> {
        let attenuation = Colour {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        let refraction_ratio = if hit_record.is_front_face {
            1.0 / self.index_of_refraction
        } else {
            self.index_of_refraction
        };

        let unit_direction = ray_in.direction.unit_vector();

        let cos_theta = f32::min(dot(unit_direction.multiply(-1.0), hit_record.normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = (refraction_ratio * sin_theta) > 1.0;

        let direction =
            if cannot_refract || reflectance(cos_theta, refraction_ratio) > random_unit() {
                reflect(unit_direction, hit_record.normal)
            } else {
                refract(unit_direction, hit_record.normal, refraction_ratio)
            };

        let scattered_ray = Ray {
            origin: hit_record.p,
            direction,
        };

        return Some((scattered_ray, attenuation));
    }
}

// Schlick's approximation for reflectance - this wasn't even explained!
fn reflectance(cosine: f32, refraction_ratio: f32) -> f32 {
    let r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
    let r0_squared = r0 * r0;
    return r0_squared + (1.0 - r0) * (1.0 - cosine).powi(5);
}

// Get the reflection off vector after hitting a surface with unit normal
// surface_normal.
// See https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection
fn reflect(incoming_ray: Vec3, surface_normal: Vec3) -> Vec3 {
    let length_of_b = dot(incoming_ray, surface_normal);

    // Don't really understand the signs of
    // In the diagram the reflection is v + 2B
    // from whence v - 2B?
    // Oh, likely explanation: The dot product gives us a negative value for
    // the length of B. We could alternatively take abs value of this.
    return incoming_ray.subtract(&surface_normal.multiply(2.0 * length_of_b));
}
