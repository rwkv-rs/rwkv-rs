pub struct Sampler {
    pub num_devices: u64,
    pub device_index: u64,
    pub samples_per_epoch: u64,
    pub magic_prime: u64,
}

impl Sampler {
    pub fn new(
        num_devices: u64,
        device_index: u64,
        samples_per_epoch: u64,
        magic_prime: u64,
    ) -> Self {
        Self {
            num_devices,
            device_index,
            samples_per_epoch,
            magic_prime,
        }
    }

    pub fn get_base_offset(&self, index: u64, mini_epoch_index: u64) -> u64 {
        let distributed_idx = index * self.num_devices + self.device_index;

        let unique_sample_index = 1
            + mini_epoch_index * self.samples_per_epoch
            + (distributed_idx * self.num_devices)
            + self.device_index;

        let u_mod_p = unique_sample_index % self.magic_prime;

        let u2_mod_p = (u_mod_p * u_mod_p) % self.magic_prime;

        let u3_mod_p = (u2_mod_p * u_mod_p) % self.magic_prime;

        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;

        (u3_mod_p * ((self.magic_prime as f64) * phi).floor() as u64) % self.magic_prime
    }
}

pub fn calculate_magic_prime(num_slots: u64) -> u64 {
    (0..num_slots)
        .rev()
        .find(|&x| x % 3 == 2 && primes::is_prime(x))
        .unwrap_or(0)
}
