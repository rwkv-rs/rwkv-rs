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
        let unique_sample_index = 1
            + mini_epoch_index * self.samples_per_epoch
            + (index * self.num_devices)
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

#[cfg(test)]
mod tests {
    use super::Sampler;
    use crate::mmap::{bin, bin_old};
    use rwkv_config::{DatasetFormatOptions, load_toml, raw::train::RawTrainConfig};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::process::Command;

    fn parse_meta(line: &str) -> HashMap<String, u64> {
        let mut map = HashMap::new();
        let line = line.strip_prefix("META ").expect("missing META line");
        for part in line.split_whitespace() {
            let (key, value) = part
                .split_once('=')
                .unwrap_or_else(|| panic!("invalid meta entry: {part}"));
            let parsed = value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid meta value for {key}: {value}"));
            map.insert(key.to_string(), parsed);
        }
        map
    }

    fn parse_offsets(output: &str) -> (HashMap<String, u64>, Vec<u64>) {
        let mut lines = output.lines();
        let meta_line = lines
            .next()
            .unwrap_or_else(|| panic!("missing META line in python output"));
        let meta = parse_meta(meta_line);
        let offsets_header = lines
            .next()
            .unwrap_or_else(|| panic!("missing OFFSETS header in python output"));
        assert_eq!(offsets_header.trim(), "OFFSETS");
        let offsets = lines
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.trim()
                    .parse::<u64>()
                    .unwrap_or_else(|_| panic!("invalid offset value: {line}"))
            })
            .collect::<Vec<_>>();
        (meta, offsets)
    }

    fn python_offsets(
        script_path: &str,
        data_prefix: &str,
        ctx_len: u64,
        magic_prime: u64,
        num_devices: usize,
        device_index: usize,
        micro_bsz: usize,
        epoch: u64,
        num_nodes: usize,
        vocab_size: u64,
    ) -> (HashMap<String, u64>, Vec<u64>) {
        let python_user_base = "/tmp/pyuser";
        let python_path = "/tmp/pyuser/lib/python3.12/site-packages";
        let output = Command::new("python3")
            .env("PYTHONUSERBASE", python_user_base)
            .env("PYTHONPATH", python_path)
            .arg(script_path)
            .arg("--data-file")
            .arg(data_prefix)
            .arg("--ctx-len")
            .arg(ctx_len.to_string())
            .arg("--magic-prime")
            .arg(magic_prime.to_string())
            .arg("--num-devices")
            .arg(num_devices.to_string())
            .arg("--device-index")
            .arg(device_index.to_string())
            .arg("--micro-bsz")
            .arg(micro_bsz.to_string())
            .arg("--epoch")
            .arg(epoch.to_string())
            .arg("--num-nodes")
            .arg(num_nodes.to_string())
            .arg("--vocab-size")
            .arg(vocab_size.to_string())
            .output()
            .expect("failed to execute python3");

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("python script failed.\nstdout:\n{stdout}\nstderr:\n{stderr}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_offsets(&stdout)
    }

    #[test]
    fn compare_with_v7_python_sampling() {
        let test_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let mut raw =
            load_toml::<_, RawTrainConfig>(test_root.join("examples/rwkv-lm/config/train.toml"));
        raw.fill_default();

        let dataset_format = raw
            .dataset_format
            .unwrap_or(DatasetFormatOptions::Rwkv);
        let dataset_base_path = raw.dataset_base_path;
        let filename = raw.filename_without_extensions;
        let data_prefix = PathBuf::from(&dataset_base_path).join(&filename);
        let bin_path = data_prefix.with_extension("bin");

        let ctx_len = raw.context_length.unwrap_or(256) as u64;
        let num_nodes = raw.num_nodes.unwrap_or(1);
        let micro_bsz = raw.batch_size_per_device.unwrap_or(1);
        let vocab_size = 0u64;

        let magic_prime = match dataset_format {
            DatasetFormatOptions::RwkvLegacy => {
                let reader = bin_old::BinReader::<u16>::new(&bin_path);
                reader.get_magic_prime(ctx_len)
            }
            DatasetFormatOptions::Rwkv => {
                let reader = bin::BinReader::<u16>::new(&bin_path);
                reader.get_magic_prime(ctx_len)
            }
        };

        let script_path = test_root.join("tmp/compare_rwkv_v7_sampling.py");
        assert!(
            script_path.exists(),
            "missing python script at {}",
            script_path.display()
        );
        let epoch = 0u64;
        let device_counts = [1usize, 4usize, 6usize];

        for &num_devices in &device_counts {
            let real_bsz = num_nodes * num_devices * micro_bsz;
            let epoch_steps = 40320 / real_bsz * num_devices;
            assert_eq!(epoch_steps * real_bsz, 40320 * num_devices);
            let samples_per_epoch = epoch_steps * real_bsz;
            let per_device_len = epoch_steps;

            println!(
                "Devices={num_devices} per_device_len={per_device_len} samples_per_epoch={samples_per_epoch}"
            );

            for device_index in 0..num_devices {
                let (meta, offsets_py) = python_offsets(
                    script_path.to_string_lossy().as_ref(),
                    data_prefix.to_string_lossy().as_ref(),
                    ctx_len,
                    magic_prime,
                    num_devices,
                    device_index,
                    micro_bsz,
                    epoch,
                    num_nodes,
                    vocab_size,
                );

                assert_eq!(
                    meta.get("samples_per_epoch"),
                    Some(&(samples_per_epoch as u64))
                );
                assert_eq!(
                    meta.get("per_device_len"),
                    Some(&(per_device_len as u64))
                );
                assert_eq!(
                    meta.get("epoch_steps"),
                    Some(&(epoch_steps as u64))
                );
                assert_eq!(meta.get("real_bsz"), Some(&(real_bsz as u64)));
                assert_eq!(meta.get("ctx_len"), Some(&ctx_len));
                assert_eq!(meta.get("magic_prime"), Some(&magic_prime));
                assert_eq!(
                    meta.get("num_devices"),
                    Some(&(num_devices as u64))
                );
                assert_eq!(
                    meta.get("device_index"),
                    Some(&(device_index as u64))
                );
                assert_eq!(meta.get("epoch"), Some(&epoch));

                assert_eq!(offsets_py.len(), per_device_len);

                let sampler = Sampler::new(
                    num_devices as u64,
                    device_index as u64,
                    samples_per_epoch as u64,
                    magic_prime,
                );

                for (idx, &offset_py) in offsets_py.iter().enumerate() {
                    let offset_rs = sampler.get_base_offset(idx as u64, epoch);
                    assert!(
                        offset_rs < magic_prime,
                        "offset out of range: {offset_rs} >= {magic_prime}"
                    );
                    assert_eq!(
                        offset_rs, offset_py,
                        "mismatch at device {device_index}, idx {idx}"
                    );
                }

                let preview = offsets_py.iter().take(5).collect::<Vec<_>>();
                println!(
                    "  device={device_index} offsets[0..5]={preview:?}"
                );
            }
        }
    }
}
