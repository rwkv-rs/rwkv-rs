use std::{fs, path::Path};

use chrono::{Datelike, Duration, NaiveDate};
use sonic_rs::{Object as Map, Value, json, prelude::*};

use super::{
    EnvAssertion,
    EnvFunctionCall,
    FunctionCall,
    TauDomainEnv,
    ToolArgSpec,
    ToolRequestor,
    ToolSpec,
    as_array,
    as_array_mut,
    as_object,
    as_object_mut,
    get_bool_field,
    get_f64_field,
    get_string_field,
    get_value,
    update_json,
};

const TODAY: &str = "2025-02-25";

const TELECOM_ASSISTANT_TOOLS: &[ToolSpec] = &[
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_customer_by_phone",
        description: "Look up a telecom customer by phone number.",
        arguments: &[ToolArgSpec {
            name: "phone_number",
            description: "customer phone number",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_customer_by_id",
        description: "Fetch a telecom customer by customer id.",
        arguments: &[ToolArgSpec {
            name: "customer_id",
            description: "customer id",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_customer_by_name",
        description: "Search customers by full name and date of birth.",
        arguments: &[
            ToolArgSpec {
                name: "full_name",
                description: "full customer name",
            },
            ToolArgSpec {
                name: "dob",
                description: "date of birth in YYYY-MM-DD",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_details_by_id",
        description: "Fetch a plan, device, line, bill, or customer by id.",
        arguments: &[ToolArgSpec {
            name: "id",
            description: "entity id",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_bills_for_customer",
        description: "List recent customer bills.",
        arguments: &[ToolArgSpec {
            name: "customer_id",
            description: "customer id",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_data_usage",
        description: "Fetch current line data usage, limit, and refueling amount.",
        arguments: &[
            ToolArgSpec {
                name: "customer_id",
                description: "customer id",
            },
            ToolArgSpec {
                name: "line_id",
                description: "line id",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "resume_line",
        description: "Resume a suspended line.",
        arguments: &[
            ToolArgSpec {
                name: "customer_id",
                description: "customer id",
            },
            ToolArgSpec {
                name: "line_id",
                description: "line id",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "send_payment_request",
        description: "Send a payment request for a bill.",
        arguments: &[
            ToolArgSpec {
                name: "customer_id",
                description: "customer id",
            },
            ToolArgSpec {
                name: "bill_id",
                description: "bill id",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "enable_roaming",
        description: "Enable roaming on a line.",
        arguments: &[
            ToolArgSpec {
                name: "customer_id",
                description: "customer id",
            },
            ToolArgSpec {
                name: "line_id",
                description: "line id",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "disable_roaming",
        description: "Disable roaming on a line.",
        arguments: &[
            ToolArgSpec {
                name: "customer_id",
                description: "customer id",
            },
            ToolArgSpec {
                name: "line_id",
                description: "line id",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "refuel_data",
        description: "Add paid data refueling to a line.",
        arguments: &[
            ToolArgSpec {
                name: "customer_id",
                description: "customer id",
            },
            ToolArgSpec {
                name: "line_id",
                description: "line id",
            },
            ToolArgSpec {
                name: "gb_amount",
                description: "gigabytes to add",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "transfer_to_human_agents",
        description: "Transfer the case to a human agent.",
        arguments: &[ToolArgSpec {
            name: "summary",
            description: "brief issue summary",
        }],
    },
];

const TELECOM_USER_TOOLS: &[ToolSpec] = &[
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_status_bar",
        description: "Inspect the user's phone status bar indicators.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_network_status",
        description: "Inspect cellular and Wi-Fi connection status.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_network_mode_preference",
        description: "Check the preferred network mode on the device.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "set_network_mode_preference",
        description: "Change the preferred network mode on the device.",
        arguments: &[ToolArgSpec {
            name: "mode",
            description: "one of 4g_5g_preferred, 4g_only, 3g_only, 2g_only",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "run_speed_test",
        description: "Run a mobile-data speed test on the device.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "toggle_airplane_mode",
        description: "Toggle airplane mode on the device.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_sim_status",
        description: "Inspect the SIM card state.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "reseat_sim_card",
        description: "Re-seat the SIM card.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "toggle_data",
        description: "Toggle mobile data.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "toggle_roaming",
        description: "Toggle the phone's roaming setting.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_data_restriction_status",
        description: "Inspect whether data saver is enabled.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "toggle_data_saver_mode",
        description: "Toggle data saver mode.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_apn_settings",
        description: "Inspect the current APN configuration.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "reset_apn_settings",
        description: "Mark APN settings to reset on reboot.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_wifi_calling_status",
        description: "Inspect Wi-Fi calling state.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "toggle_wifi_calling",
        description: "Toggle Wi-Fi calling.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_vpn_status",
        description: "Inspect VPN status.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "disconnect_vpn",
        description: "Disconnect the active VPN.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_app_status",
        description: "Inspect an app's current permissions.",
        arguments: &[ToolArgSpec {
            name: "app_name",
            description: "app name",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_app_permissions",
        description: "Inspect an app's permission set.",
        arguments: &[ToolArgSpec {
            name: "app_name",
            description: "app name",
        }],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "grant_app_permission",
        description: "Grant a permission to an app.",
        arguments: &[
            ToolArgSpec {
                name: "app_name",
                description: "app name",
            },
            ToolArgSpec {
                name: "permission",
                description: "permission name",
            },
        ],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "can_send_mms",
        description: "Check whether the device can send MMS.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "reboot_device",
        description: "Reboot the user's device.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "check_payment_request",
        description: "Check whether the user has a pending payment request.",
        arguments: &[],
    },
    ToolSpec {
        requestor: ToolRequestor::User,
        name: "make_payment",
        description: "Pay the currently pending payment request.",
        arguments: &[],
    },
];

pub struct TelecomEnv {
    policy: String,
    db: Value,
    user_db: Value,
}

impl TelecomEnv {
    pub fn load(dataset_root: &Path) -> Result<Self, String> {
        let root = dataset_root.join("tau_bench").join("telecom");
        let db = parse_toml_value(&root.join("db.toml"))?;
        let mut user_db = parse_toml_value(&root.join("user_db.toml"))?;
        ensure_surroundings(&mut user_db)?;
        let main_policy =
            fs::read_to_string(root.join("main_policy.md")).map_err(|err| err.to_string())?;
        let tech_support = fs::read_to_string(root.join("tech_support_manual.md"))
            .map_err(|err| err.to_string())?;
        let policy = format!(
            "<main_policy>\n{}\n</main_policy>\n<tech_support_policy>\n{}\n</tech_support_policy>",
            main_policy.trim(),
            tech_support.trim()
        );
        let mut env = Self {
            policy,
            db,
            user_db,
        };
        env.sync_tools()?;
        Ok(env)
    }

    fn sync_tools(&mut self) -> Result<(), String> {
        let phone_number = self
            .surroundings()
            .get(&"phone_number")
            .and_then(Value::as_str)
            .map(str::to_string);
        if phone_number.is_none() {
            return Ok(());
        }
        let phone_number = phone_number.unwrap();
        let line_id = self.find_line_index_by_phone(&phone_number)?;
        let customer_idx = self.find_customer_index_by_phone_or_line(&phone_number)?;

        let line = self.db_line(line_id)?.clone();
        let plan = self
            .plan_by_id(get_string_field(as_object(&line)?, "plan_id")?)?
            .clone();
        {
            let surroundings = self.surroundings_mut()?;
            surroundings.insert(
                &"line_active",
                json!(get_string_field(as_object(&line)?, "status")? == "Active"),
            );
            surroundings.insert(
                &"roaming_allowed",
                json!(get_bool_field(as_object(&line)?, "roaming_enabled")?),
            );
            let data_used = get_f64_field(as_object(&line)?, "data_used_gb")?;
            let refueling = get_f64_field(as_object(&line)?, "data_refueling_gb")?;
            let limit = get_f64_field(as_object(&plan)?, "data_limit_gb")?;
            surroundings.insert(
                &"mobile_data_usage_exceeded",
                json!(data_used >= limit + refueling),
            );
        }

        let paid_request = self
            .surroundings()
            .get(&"payment_request")
            .and_then(Value::as_object)
            .is_some_and(|request| {
                request
                    .get(&"paid")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            });
        if paid_request {
            if let Some(bill_id) = self
                .surroundings()
                .get(&"payment_request")
                .and_then(Value::as_object)
                .and_then(|request| request.get(&"bill_id"))
                .and_then(Value::as_str)
                .map(str::to_string)
            {
                self.set_bill_to_paid(&bill_id)?;
            }
            self.surroundings_mut()?
                .insert(&"payment_request", Value::new_null());
        }

        let customer = self.db_customer(customer_idx)?.clone();
        if self
            .surroundings()
            .get(&"payment_request")
            .is_none_or(Value::is_null)
        {
            if let Some(bill) = self.first_awaiting_payment_bill(&customer)? {
                self.surroundings_mut()?.insert(
                    &"payment_request",
                    json!({
                        "bill_id": get_string_field(as_object(&bill)?, "bill_id")?,
                        "amount_due": get_f64_field(as_object(&bill)?, "total_due")?,
                        "paid": false,
                    }),
                );
            }
        }

        self.simulate_network_search()?;
        Ok(())
    }

    fn execute_assistant_tool(&mut self, name: &str, args: &Map) -> Result<Value, String> {
        match name {
            "get_customer_by_phone" => {
                let phone_number = str_arg(args, "phone_number")?;
                let idx = self.find_customer_index_by_phone_or_line(phone_number)?;
                Ok(self.db_customer(idx)?.clone())
            }
            "get_customer_by_id" => {
                let idx = self.find_customer_index_by_id(str_arg(args, "customer_id")?)?;
                Ok(self.db_customer(idx)?.clone())
            }
            "get_customer_by_name" => {
                let full_name = str_arg(args, "full_name")?.to_lowercase();
                let dob = str_arg(args, "dob")?;
                let customers = self.db_customers()?;
                Ok(json!(
                    customers
                        .iter()
                        .filter(|customer| {
                            let object = customer.as_object().unwrap();
                            object
                                .get(&"full_name")
                                .and_then(Value::as_str)
                                .is_some_and(|value| value.to_lowercase() == full_name)
                                && object.get(&"date_of_birth").and_then(Value::as_str) == Some(dob)
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                ))
            }
            "get_details_by_id" => self.get_details_by_id(str_arg(args, "id")?),
            "get_bills_for_customer" => self.get_bills_for_customer(str_arg(args, "customer_id")?),
            "get_data_usage" => {
                self.get_data_usage(str_arg(args, "customer_id")?, str_arg(args, "line_id")?)
            }
            "resume_line" => {
                self.resume_line(str_arg(args, "customer_id")?, str_arg(args, "line_id")?)
            }
            "send_payment_request" => {
                self.send_payment_request(str_arg(args, "customer_id")?, str_arg(args, "bill_id")?)
            }
            "enable_roaming" => {
                self.enable_roaming(str_arg(args, "customer_id")?, str_arg(args, "line_id")?)
            }
            "disable_roaming" => {
                self.disable_roaming(str_arg(args, "customer_id")?, str_arg(args, "line_id")?)
            }
            "refuel_data" => self.refuel_data(
                str_arg(args, "customer_id")?,
                str_arg(args, "line_id")?,
                num_arg(args, "gb_amount")?,
            ),
            "transfer_to_human_agents" => Ok(json!("Transfer successful".to_string())),
            _ => Err(format!("unknown assistant telecom tool `{name}`")),
        }
    }

    fn execute_user_tool(&mut self, name: &str, args: &Map) -> Result<Value, String> {
        match name {
            "check_status_bar" => Ok(json!(self.check_status_bar()?)),
            "check_network_status" => Ok(json!(self.check_network_status()?)),
            "check_network_mode_preference" => Ok(json!(self.check_network_mode_preference()?)),
            "set_network_mode_preference" => Ok(json!(
                self.set_network_mode_preference(str_arg(args, "mode")?)?
            )),
            "run_speed_test" => Ok(json!(self.run_speed_test()?)),
            "toggle_airplane_mode" => Ok(json!(self.toggle_airplane_mode()?)),
            "check_sim_status" => Ok(json!(self.check_sim_status()?)),
            "reseat_sim_card" => Ok(json!(self.reseat_sim_card()?)),
            "toggle_data" => Ok(json!(self.toggle_data()?)),
            "toggle_roaming" => Ok(json!(self.toggle_roaming()?)),
            "check_data_restriction_status" => Ok(json!(self.check_data_restriction_status()?)),
            "toggle_data_saver_mode" => Ok(json!(self.toggle_data_saver_mode()?)),
            "check_apn_settings" => Ok(json!(self.check_apn_settings()?)),
            "reset_apn_settings" => Ok(json!(self.reset_apn_settings()?)),
            "check_wifi_calling_status" => Ok(json!(self.check_wifi_calling_status()?)),
            "toggle_wifi_calling" => Ok(json!(self.toggle_wifi_calling()?)),
            "check_vpn_status" => Ok(json!(self.check_vpn_status()?)),
            "disconnect_vpn" => Ok(json!(self.disconnect_vpn()?)),
            "check_app_status" => Ok(json!(self.check_app_status(str_arg(args, "app_name")?)?)),
            "check_app_permissions" => Ok(json!(
                self.check_app_permissions(str_arg(args, "app_name")?)?
            )),
            "grant_app_permission" => {
                Ok(json!(self.grant_app_permission(
                    str_arg(args, "app_name")?,
                    str_arg(args, "permission")?,
                )?))
            }
            "can_send_mms" => Ok(json!(self.can_send_mms()?)),
            "reboot_device" => Ok(json!(self.reboot_device()?)),
            "check_payment_request" => Ok(json!(self.check_payment_request()?)),
            "make_payment" => Ok(json!(self.make_payment()?)),
            _ => Err(format!("unknown user telecom tool `{name}`")),
        }
    }

    fn get_details_by_id(&self, id: &str) -> Result<Value, String> {
        if id.starts_with('L') {
            Ok(self.find_line_by_id(id)?.clone())
        } else if id.starts_with('D') {
            Ok(self.find_device_by_id(id)?.clone())
        } else if id.starts_with('B') {
            Ok(self.find_bill_by_id(id)?.clone())
        } else if id.starts_with('C') {
            Ok(self.find_customer_by_id(id)?.clone())
        } else if id.starts_with('P') {
            Ok(self.plan_by_id(id)?.clone())
        } else {
            Err(format!("unknown id format: {id}"))
        }
    }

    fn get_bills_for_customer(&self, customer_id: &str) -> Result<Value, String> {
        let customer = self.find_customer_by_id(customer_id)?;
        let bill_ids = get_value(as_object(customer)?, "bill_ids")?
            .as_array()
            .ok_or_else(|| "customer bill_ids must be an array".to_string())?;
        let mut bills = bill_ids
            .iter()
            .filter_map(Value::as_str)
            .map(|bill_id| self.find_bill_by_id(bill_id).map(Clone::clone))
            .collect::<Result<Vec<_>, _>>()?;
        bills.sort_by_key(|bill| {
            bill.get(&"issue_date")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        });
        bills.reverse();
        Ok(json!(bills))
    }

    fn get_data_usage(&self, customer_id: &str, line_id: &str) -> Result<Value, String> {
        self.ensure_customer_has_line(customer_id, line_id)?;
        let line = self.find_line_by_id(line_id)?;
        let plan = self.plan_by_id(get_string_field(as_object(line)?, "plan_id")?)?;
        let today = parse_date(TODAY)?;
        let next_month = if today.month() == 12 {
            NaiveDate::from_ymd_opt(today.year() + 1, 1, 1).unwrap()
        } else {
            NaiveDate::from_ymd_opt(today.year(), today.month() + 1, 1).unwrap()
        };
        let cycle_end_date = next_month - Duration::days(1);
        Ok(json!({
            "line_id": line_id,
            "data_used_gb": get_f64_field(as_object(line)?, "data_used_gb")?,
            "data_limit_gb": get_f64_field(as_object(plan)?, "data_limit_gb")?,
            "data_refueling_gb": get_f64_field(as_object(line)?, "data_refueling_gb")?,
            "cycle_end_date": cycle_end_date.to_string(),
        }))
    }

    fn resume_line(&mut self, customer_id: &str, line_id: &str) -> Result<Value, String> {
        self.ensure_customer_has_line(customer_id, line_id)?;
        let idx = self.find_line_index_by_id(line_id)?;
        let line = {
            let line = self.db_line_mut(idx)?;
            let status = get_string_field(as_object(line)?, "status")?;
            if status != "Suspended" && status != "Pending Activation" {
                return Err("Line must be suspended to resume".to_string());
            }
            as_object_mut(line)?.insert(&"status", json!("Active".to_string()));
            as_object_mut(line)?.insert(&"suspension_start_date", Value::new_null());
            line.clone()
        };
        self.sync_tools()?;
        Ok(json!({
            "message": "Line resumed successfully",
            "line": line,
        }))
    }

    fn send_payment_request(&mut self, customer_id: &str, bill_id: &str) -> Result<Value, String> {
        let customer = self.find_customer_by_id(customer_id)?.clone();
        if !as_array(get_value(as_object(&customer)?, "bill_ids")?)?
            .iter()
            .filter_map(Value::as_str)
            .any(|candidate| candidate == bill_id)
        {
            return Err(format!(
                "Bill {bill_id} not found for customer {customer_id}"
            ));
        }
        let awaiting = self.awaiting_payment_bills(&customer)?;
        if !awaiting.is_empty() {
            return Err("A bill is already awaiting payment for this customer".to_string());
        }
        let bill_idx = self.find_bill_index_by_id(bill_id)?;
        as_object_mut(self.db_bill_mut(bill_idx)?)?
            .insert(&"status", json!("Awaiting Payment".to_string()));
        self.sync_tools()?;
        Ok(json!(format!(
            "Payment request sent to the customer for bill {bill_id}"
        )))
    }

    fn enable_roaming(&mut self, customer_id: &str, line_id: &str) -> Result<Value, String> {
        self.ensure_customer_has_line(customer_id, line_id)?;
        let idx = self.find_line_index_by_id(line_id)?;
        let line = self.db_line_mut(idx)?;
        let already = get_bool_field(as_object(line)?, "roaming_enabled")?;
        if already {
            return Ok(json!("Roaming was already enabled".to_string()));
        }
        as_object_mut(line)?.insert(&"roaming_enabled", json!(true));
        self.sync_tools()?;
        Ok(json!("Roaming enabled successfully".to_string()))
    }

    fn disable_roaming(&mut self, customer_id: &str, line_id: &str) -> Result<Value, String> {
        self.ensure_customer_has_line(customer_id, line_id)?;
        let idx = self.find_line_index_by_id(line_id)?;
        let line = self.db_line_mut(idx)?;
        let already = !get_bool_field(as_object(line)?, "roaming_enabled")?;
        if already {
            return Ok(json!("Roaming was already disabled".to_string()));
        }
        as_object_mut(line)?.insert(&"roaming_enabled", json!(false));
        self.sync_tools()?;
        Ok(json!("Roaming disabled successfully".to_string()))
    }

    fn refuel_data(
        &mut self,
        customer_id: &str,
        line_id: &str,
        gb_amount: f64,
    ) -> Result<Value, String> {
        self.ensure_customer_has_line(customer_id, line_id)?;
        if gb_amount <= 0.0 {
            return Err("Refuel amount must be positive".to_string());
        }
        let line_idx = self.find_line_index_by_id(line_id)?;
        let plan_id = {
            let line = self.db_line(line_idx)?;
            get_string_field(as_object(line)?, "plan_id")?.to_string()
        };
        let plan = self.plan_by_id(&plan_id)?.clone();
        let charge_amount =
            gb_amount * get_f64_field(as_object(&plan)?, "data_refueling_price_per_gb")?;
        {
            let line = self.db_line_mut(line_idx)?;
            let current = get_f64_field(as_object(line)?, "data_refueling_gb")?;
            as_object_mut(line)?.insert(
                &"data_refueling_gb",
                json!(((current + gb_amount) * 100.0).round() / 100.0),
            );
        }
        self.apply_one_time_charge(
            customer_id,
            charge_amount,
            format!(
                "Data refueling: {gb_amount} GB at ${}/GB",
                get_f64_field(as_object(&plan)?, "data_refueling_price_per_gb")?
            ),
        )?;
        self.sync_tools()?;
        let new_amount = get_f64_field(
            as_object(self.find_line_by_id(line_id)?)?,
            "data_refueling_gb",
        )?;
        Ok(json!({
            "message": format!("Successfully added {gb_amount} GB of data for line {line_id} for ${charge_amount:.2}"),
            "new_data_refueling_gb": new_amount,
            "charge": charge_amount,
        }))
    }

    fn apply_one_time_charge(
        &mut self,
        customer_id: &str,
        amount: f64,
        description: String,
    ) -> Result<(), String> {
        let customer_idx = self.find_customer_index_by_id(customer_id)?;
        let bill_ids = {
            let customer = self.db_customer(customer_idx)?;
            as_array(get_value(as_object(customer)?, "bill_ids")?)?.clone()
        };
        let mut draft_idx = None;
        for bill_id in bill_ids.iter().filter_map(Value::as_str) {
            let idx = self.find_bill_index_by_id(bill_id)?;
            if get_string_field(as_object(self.db_bill(idx)?)?, "status")? == "Draft" {
                draft_idx = Some(idx);
                break;
            }
        }
        if draft_idx.is_none() {
            let today = parse_date(TODAY)?;
            let next_month = if today.month() == 12 {
                NaiveDate::from_ymd_opt(today.year() + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(today.year(), today.month() + 1, 1).unwrap()
            };
            let period_end = if next_month.month() == 12 {
                NaiveDate::from_ymd_opt(next_month.year() + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(next_month.year(), next_month.month() + 1, 1).unwrap()
            } - Duration::days(1);
            let new_bill_id = format!(
                "B{}",
                9000 + as_array(get_value(as_object(&self.db)?, "bills")?)?.len()
            );
            as_array_mut(get_value_mut(as_object_mut(&mut self.db)?, "bills")?)?.push(json!({
                "bill_id": new_bill_id,
                "customer_id": customer_id,
                "period_start": next_month.to_string(),
                "period_end": period_end.to_string(),
                "issue_date": next_month.to_string(),
                "total_due": 0.0,
                "due_date": (next_month + Duration::days(14)).to_string(),
                "status": "Draft",
                "line_items": [],
            }));
            as_array_mut(get_value_mut(
                as_object_mut(self.db_customer_mut(customer_idx)?)?,
                "bill_ids",
            )?)?
            .push(json!(new_bill_id.clone()));
            draft_idx = Some(self.find_bill_index_by_id(&new_bill_id)?);
        }
        let draft_idx = draft_idx.unwrap();
        let bill = self.db_bill_mut(draft_idx)?;
        as_array_mut(get_value_mut(as_object_mut(bill)?, "line_items")?)?.push(json!({
            "description": description,
            "amount": amount,
            "date": TODAY,
            "item_type": if amount < 0.0 { "Credit" } else { "Charge" },
        }));
        let total_due = get_f64_field(as_object(bill)?, "total_due")?;
        as_object_mut(bill)?.insert(&"total_due", json!(total_due + amount));
        Ok(())
    }

    fn check_status_bar(&self) -> Result<String, String> {
        let device = self.device()?;
        let mut indicators = Vec::new();
        if get_bool_field(device, "airplane_mode")? {
            indicators.push("Airplane Mode".to_string());
        } else {
            indicators.push(
                match get_string_field(device, "network_signal_strength")? {
                    "none" => "No Signal",
                    "poor" => "Poor Signal",
                    "fair" => "Fair Signal",
                    "good" => "Good Signal",
                    "excellent" => "Excellent Signal",
                    other => other,
                }
                .to_string(),
            );
            let network = get_string_field(device, "network_technology_connected")?;
            if network != "none" {
                indicators.push(network.to_string());
            }
            indicators.push(
                if get_bool_field(device, "data_enabled")? && network != "none" {
                    "Data Enabled".to_string()
                } else {
                    "Data Disabled".to_string()
                },
            );
            if get_bool_field(device, "data_saver_mode")? {
                indicators.push("Data Saver".to_string());
            }
        }
        if get_bool_field(device, "wifi_enabled")? && get_bool_field(device, "wifi_connected")? {
            indicators.push(format!(
                "Wi-Fi {}",
                device
                    .get(&"wifi_ssid")
                    .and_then(Value::as_str)
                    .unwrap_or("Connected")
            ));
        }
        if get_bool_field(device, "vpn_connected")? {
            indicators.push("VPN Connected".to_string());
        }
        indicators.push(format!(
            "Battery {}%",
            device
                .get(&"battery_level")
                .and_then(Value::as_i64)
                .unwrap_or(0)
        ));
        Ok(indicators.join(" | "))
    }

    fn check_network_status(&self) -> Result<String, String> {
        let device = self.device()?;
        Ok(format!(
            "Airplane Mode: {}\nSIM Card Status: {}\nCellular Connection: {}\nCellular Signal: {}\nCellular Network Type: {}\nMobile Data Enabled: {}\nData Roaming Enabled: {}\nWi-Fi Radio: {}\nWi-Fi Connected: {}",
            yes_no(get_bool_field(device, "airplane_mode")?),
            get_string_field(device, "sim_card_status")?,
            get_string_field(device, "network_connection_status")?,
            get_string_field(device, "network_signal_strength")?,
            get_string_field(device, "network_technology_connected")?,
            yes_no(get_bool_field(device, "data_enabled")?),
            yes_no(get_bool_field(device, "roaming_enabled")?),
            yes_no(get_bool_field(device, "wifi_enabled")?),
            yes_no(get_bool_field(device, "wifi_connected")?),
        ))
    }

    fn check_network_mode_preference(&self) -> Result<String, String> {
        Ok(format!(
            "Network Mode Preference: {}",
            get_string_field(self.device()?, "network_mode_preference")?
        ))
    }

    fn set_network_mode_preference(&mut self, mode: &str) -> Result<String, String> {
        let valid = ["4g_5g_preferred", "4g_only", "3g_only", "2g_only"];
        if !valid.contains(&mode) {
            return Ok(format!(
                "Failed to set network mode: '{mode}' is not a valid option."
            ));
        }
        self.device_mut()?
            .insert(&"network_mode_preference", json!(mode.to_string()));
        self.simulate_network_search()?;
        Ok(format!("Preferred Network Mode set to: {mode}"))
    }

    fn run_speed_test(&self) -> Result<String, String> {
        let (speed, description) = self.speed_test();
        match speed {
            Some(speed) => Ok(format!(
                "Speed Test Result: {speed:.2} Mbps ({description})."
            )),
            None => Ok(format!("Speed test failed: {description}.")),
        }
    }

    fn toggle_airplane_mode(&mut self) -> Result<String, String> {
        let current = get_bool_field(self.device()?, "airplane_mode")?;
        self.device_mut()?.insert(&"airplane_mode", json!(!current));
        if !current {
            self.device_mut()?.insert(&"wifi_connected", json!(false));
            self.device_mut()?.insert(&"wifi_ssid", Value::new_null());
        }
        if current && get_bool_field(self.device()?, "wifi_enabled")? {
            self.device_mut()?.insert(&"wifi_connected", json!(false));
        }
        if !current {
            self.disconnect_vpn()?;
        }
        self.simulate_network_search()?;
        Ok(format!("Airplane Mode is now {}.", yes_no(!current)))
    }

    fn check_sim_status(&self) -> Result<String, String> {
        let status = if get_bool_field(self.device()?, "sim_card_missing")? {
            "missing"
        } else {
            get_string_field(self.device()?, "sim_card_status")?
        };
        Ok(match status {
            "active" => "Your SIM card is active and working.",
            "missing" => "No SIM card detected in the phone.",
            "locked_pin" => "The SIM card is locked with a PIN code.",
            "locked_puk" => "The SIM card is locked with a PUK code.",
            other => other,
        }
        .to_string())
    }

    fn reseat_sim_card(&mut self) -> Result<String, String> {
        self.device_mut()?.insert(&"sim_card_missing", json!(false));
        self.simulate_network_search()?;
        Ok("SIM card re-seated successfully.".to_string())
    }

    fn toggle_data(&mut self) -> Result<String, String> {
        let current = get_bool_field(self.device()?, "data_enabled")?;
        self.device_mut()?.insert(&"data_enabled", json!(!current));
        self.simulate_network_search()?;
        Ok(format!("Mobile Data is now {}.", yes_no(!current)))
    }

    fn toggle_roaming(&mut self) -> Result<String, String> {
        let current = get_bool_field(self.device()?, "roaming_enabled")?;
        self.device_mut()?
            .insert(&"roaming_enabled", json!(!current));
        self.simulate_network_search()?;
        Ok(format!("Data Roaming is now {}.", yes_no(!current)))
    }

    fn check_data_restriction_status(&self) -> Result<String, String> {
        Ok(if get_bool_field(self.device()?, "data_saver_mode")? {
            "Data Saver mode is ON (limits data usage).".to_string()
        } else {
            "Data Saver mode is OFF.".to_string()
        })
    }

    fn toggle_data_saver_mode(&mut self) -> Result<String, String> {
        let current = get_bool_field(self.device()?, "data_saver_mode")?;
        self.device_mut()?
            .insert(&"data_saver_mode", json!(!current));
        Ok(format!("Data Saver Mode is now {}.", yes_no(!current)))
    }

    fn check_apn_settings(&self) -> Result<String, String> {
        let apn = self.active_apn()?;
        Ok(format!(
            "Current APN Name: {}\nMMSC URL (for picture messages): {}",
            get_string_field(apn, "apn_name")?,
            apn.get(&"mmsc_url")
                .and_then(Value::as_str)
                .unwrap_or("Not Set"),
        ))
    }

    fn reset_apn_settings(&mut self) -> Result<String, String> {
        as_object_mut(self.active_apn_mut()?)?.insert(&"reset_at_reboot", json!(true));
        Ok("APN settings will reset at reboot.".to_string())
    }

    fn check_wifi_calling_status(&self) -> Result<String, String> {
        Ok(format!(
            "Wi-Fi Calling is currently turned {}.",
            yes_no(get_bool_field(self.device()?, "wifi_calling_enabled")?)
        ))
    }

    fn toggle_wifi_calling(&mut self) -> Result<String, String> {
        let current = get_bool_field(self.device()?, "wifi_calling_enabled")?;
        self.device_mut()?
            .insert(&"wifi_calling_enabled", json!(!current));
        Ok(format!("Wi-Fi Calling is now {}.", yes_no(!current)))
    }

    fn check_vpn_status(&self) -> Result<String, String> {
        let device = self.device()?;
        if get_bool_field(device, "vpn_connected")? {
            Ok("VPN is ON and connected.".to_string())
        } else if get_bool_field(device, "vpn_enabled_setting")? {
            Ok("VPN is turned ON in settings, but currently not connected.".to_string())
        } else {
            Ok("VPN is turned OFF.".to_string())
        }
    }

    fn disconnect_vpn(&mut self) -> Result<String, String> {
        self.device_mut()?.insert(&"vpn_connected", json!(false));
        self.device_mut()?.insert(&"vpn_details", Value::new_null());
        Ok("VPN disconnected successfully.".to_string())
    }

    fn check_app_status(&self, app_name: &str) -> Result<String, String> {
        let app = match self.app_status(app_name) {
            Some(value) => value,
            None => return Ok(format!("App '{app_name}' not found on this phone.")),
        };
        let permissions = as_object(get_value(app, "permissions")?)?;
        let granted = permissions
            .iter()
            .filter_map(|(name, value)| value.as_bool().filter(|enabled| *enabled).map(|_| name))
            .collect::<Vec<_>>();
        Ok(if granted.is_empty() {
            format!("Status for App: {app_name}\n - Permissions: None granted.")
        } else {
            format!(
                "Status for App: {app_name}\n - Permissions Granted:\n   - {}",
                granted.join("\n   - ")
            )
        })
    }

    fn check_app_permissions(&self, app_name: &str) -> Result<String, String> {
        let app = match self.app_status(app_name) {
            Some(value) => value,
            None => return Ok(format!("App '{app_name}' not found on this phone.")),
        };
        let permissions = as_object(get_value(app, "permissions")?)?;
        let granted = permissions
            .iter()
            .filter_map(|(name, value)| value.as_bool().filter(|enabled| *enabled).map(|_| name))
            .collect::<Vec<_>>();
        Ok(if granted.is_empty() {
            format!("App '{app_name}' currently has no permissions granted.")
        } else {
            format!(
                "App '{app_name}' has permission for: {}.",
                granted.join(", ")
            )
        })
    }

    fn grant_app_permission(&mut self, app_name: &str, permission: &str) -> Result<String, String> {
        let app = self.app_status_mut(app_name)?;
        let permissions = as_object_mut(get_value_mut(app, "permissions")?)?;
        if !permissions.contains_key(&permission) {
            return Ok(format!(
                "Error. Permission '{permission}' not tracked for app '{app_name}'."
            ));
        }
        permissions.insert(&permission, json!(true));
        Ok(format!(
            "Success. Permission '{permission}' granted to app '{app_name}'."
        ))
    }

    fn can_send_mms(&self) -> Result<String, String> {
        Ok(if self.can_send_mms_flag()? {
            "Your messaging app can send MMS messages.".to_string()
        } else {
            "Your messaging app cannot send MMS messages.".to_string()
        })
    }

    fn reboot_device(&mut self) -> Result<String, String> {
        if self
            .active_apn()?
            .get(&"reset_at_reboot")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            *self.active_apn_mut()? = json!({
                "apn_name": "internet",
                "reset_at_reboot": false,
                "mms_apn": "mms",
                "mmsc_url": "http://mms.carrier.com/mms/wapenc",
                "mms_proxy": null,
                "mms_port": null,
            });
        }
        self.device_mut()?
            .insert(&"network_connection_status", json!("searching".to_string()));
        self.simulate_network_search()?;
        Ok("Restarting network services...".to_string())
    }

    fn check_payment_request(&self) -> Result<String, String> {
        match self.surroundings().get(&"payment_request") {
            Some(request) => {
                let Some(request) = request.as_object() else {
                    return Ok("No payment request has been made.".to_string());
                };
                Ok(format!(
                    "You have a payment request for bill {} of {} USD.",
                    get_string_field(request, "bill_id")?,
                    request
                        .get(&"amount_due")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                ))
            }
            _ => Ok("No payment request has been made.".to_string()),
        }
    }

    fn make_payment(&mut self) -> Result<String, String> {
        let request = self
            .surroundings()
            .get(&"payment_request")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| "You do not have a payment request.".to_string())?;
        self.surroundings_mut()?.insert(
            &"payment_request",
            json!({
                "bill_id": get_string_field(&request, "bill_id")?,
                "amount_due": request.get(&"amount_due").and_then(Value::as_f64).unwrap_or(0.0),
                "paid": true,
            }),
        );
        self.sync_tools()?;
        Ok(format!(
            "Payment of {} USD has been made for bill {}.",
            request
                .get(&"amount_due")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            get_string_field(&request, "bill_id")?,
        ))
    }

    fn speed_test(&self) -> (Option<f64>, &'static str) {
        if !self.mobile_data_working().unwrap_or(false) {
            return (None, "No Connection");
        }
        let device = self.device().unwrap();
        let mut base_factor = 1.0;
        if get_bool_field(device, "vpn_connected").unwrap_or(false)
            && device
                .get(&"vpn_details")
                .and_then(Value::as_object)
                .and_then(|details| details.get(&"server_performance"))
                .and_then(Value::as_str)
                == Some("poor")
        {
            base_factor = 0.1;
        }
        if get_bool_field(device, "data_saver_mode").unwrap_or(false) {
            base_factor *= 0.2;
        }
        let (min_speed, max_speed): (f64, f64) =
            match get_string_field(device, "network_technology_connected").unwrap_or("none") {
                "2G" => (0.1, 0.4),
                "3G" => (1.0, 5.0),
                "4G" => (10.0, 100.0),
                "5G" => (50.0, 500.0),
                _ => (0.0, 0.0),
            };
        let signal_factor =
            match get_string_field(device, "network_signal_strength").unwrap_or("none") {
                "poor" => 0.2,
                "fair" => 0.5,
                "good" => 0.8,
                "excellent" => 1.0,
                _ => 0.0,
            };
        let speed =
            ((min_speed + max_speed) / 2.0 * signal_factor * base_factor * 100.0).round() / 100.0;
        let desc = if speed < 1.0 {
            "Very Poor"
        } else if speed < 5.0 {
            "Poor"
        } else if speed < 25.0 {
            "Fair"
        } else if speed < 100.0 {
            "Good"
        } else {
            "Excellent"
        };
        (Some(speed), desc)
    }

    fn mobile_data_working(&self) -> Result<bool, String> {
        let device = self.device()?;
        if get_bool_field(device, "airplane_mode")?
            || get_string_field(device, "network_signal_strength")? == "none"
        {
            return Ok(false);
        }
        if get_string_field(device, "network_connection_status")? == "no_service" {
            return Ok(false);
        }
        let surroundings = self.surroundings();
        if surroundings
            .get(&"is_abroad")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && (!get_bool_field(device, "roaming_enabled")?
                || !surroundings
                    .get(&"roaming_allowed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false))
        {
            return Ok(false);
        }
        if !get_bool_field(device, "data_enabled")? {
            return Ok(false);
        }
        if surroundings
            .get(&"mobile_data_usage_exceeded")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn can_send_mms_flag(&self) -> Result<bool, String> {
        if !self.mobile_data_working()? {
            return Ok(false);
        }
        let device = self.device()?;
        if get_string_field(device, "network_technology_connected")? == "2G" {
            return Ok(false);
        }
        if get_bool_field(device, "wifi_calling_enabled")?
            && get_bool_field(device, "wifi_calling_mms_over_wifi")?
        {
            return Ok(false);
        }
        if self
            .active_apn()?
            .get(&"mmsc_url")
            .is_none_or(Value::is_null)
        {
            return Ok(false);
        }
        let permissions = self
            .app_status("messaging")
            .and_then(|app| get_value(app, "permissions").ok())
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Ok(permissions
            .get(&"storage")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && permissions
                .get(&"sms")
                .and_then(Value::as_bool)
                .unwrap_or(false))
    }

    fn simulate_network_search(&mut self) -> Result<(), String> {
        let sim_missing = get_bool_field(self.device()?, "sim_card_missing")?;
        let sim_status = if sim_missing {
            "missing".to_string()
        } else {
            get_string_field(self.device()?, "sim_card_status")?.to_string()
        };

        let (mut connection, mut technology, mut signal) = if sim_status == "active" {
            let pref = get_string_field(self.device()?, "network_mode_preference")?;
            match pref {
                "4g_5g_preferred" => {
                    let five_g = self.signal_strength_for("5G");
                    if five_g == "none" {
                        ("connected", "4G", self.signal_strength_for("4G"))
                    } else {
                        ("connected", "5G", five_g)
                    }
                }
                "4g_only" => ("connected", "4G", self.signal_strength_for("4G")),
                "3g_only" => ("connected", "3G", self.signal_strength_for("3G")),
                "2g_only" => ("connected", "2G", self.signal_strength_for("2G")),
                _ => ("connected", "4G", self.signal_strength_for("4G")),
            }
        } else {
            ("no_service", "none", "none".to_string())
        };

        if get_bool_field(self.device()?, "airplane_mode")? {
            connection = "no_service";
            technology = "none";
            signal = "none".to_string();
        }
        if get_string_field(self.active_apn()?, "apn_name")? == "broken" {
            connection = "no_service";
            technology = "none";
            signal = "none".to_string();
        }
        if !self
            .surroundings()
            .get(&"line_active")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            connection = "no_service";
            technology = "none";
            signal = "none".to_string();
        }

        self.device_mut()?
            .insert(&"network_connection_status", json!(connection.to_string()));
        self.device_mut()?.insert(
            &"network_technology_connected",
            json!(technology.to_string()),
        );
        self.device_mut()?
            .insert(&"network_signal_strength", json!(signal));
        Ok(())
    }

    fn signal_strength_for(&self, tech: &str) -> String {
        self.surroundings()
            .get(&"signal_strength")
            .and_then(Value::as_object)
            .and_then(|signals| signals.get(&tech))
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_string()
    }

    fn surroundings(&self) -> &Map {
        self.user_db
            .get(&"surroundings")
            .and_then(Value::as_object)
            .unwrap()
    }

    fn surroundings_mut(&mut self) -> Result<&mut Map, String> {
        as_object_mut(
            self.user_db
                .get_mut(&"surroundings")
                .ok_or_else(|| "missing user surroundings".to_string())?,
        )
    }

    fn device(&self) -> Result<&Map, String> {
        as_object(
            self.user_db
                .get(&"device")
                .ok_or_else(|| "missing user device".to_string())?,
        )
    }

    fn device_mut(&mut self) -> Result<&mut Map, String> {
        as_object_mut(
            self.user_db
                .get_mut(&"device")
                .ok_or_else(|| "missing user device".to_string())?,
        )
    }

    fn active_apn(&self) -> Result<&Map, String> {
        as_object(get_value(self.device()?, "active_apn_settings")?)
    }

    fn active_apn_mut(&mut self) -> Result<&mut Value, String> {
        get_value_mut(self.device_mut()?, "active_apn_settings")
    }

    fn app_status(&self, app_name: &str) -> Option<&Map> {
        self.device()
            .ok()?
            .get(&"app_statuses")?
            .as_object()?
            .get(&app_name)?
            .as_object()
    }

    fn app_status_mut(&mut self, app_name: &str) -> Result<&mut Map, String> {
        self.device_mut()?
            .get_mut(&"app_statuses")
            .and_then(Value::as_object_mut)
            .and_then(|apps| apps.get_mut(&app_name))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| format!("App '{app_name}' not found."))
    }

    fn db_customers(&self) -> Result<&sonic_rs::Array, String> {
        as_array(get_value(as_object(&self.db)?, "customers")?)
    }

    fn db_customer(&self, index: usize) -> Result<&Value, String> {
        self.db_customers()?
            .get(index)
            .ok_or_else(|| "customer index out of range".to_string())
    }

    fn db_customer_mut(&mut self, index: usize) -> Result<&mut Value, String> {
        as_array_mut(get_value_mut(as_object_mut(&mut self.db)?, "customers")?)?
            .get_mut(index)
            .ok_or_else(|| "customer index out of range".to_string())
    }

    fn db_line(&self, index: usize) -> Result<&Value, String> {
        as_array(get_value(as_object(&self.db)?, "lines")?)?
            .get(index)
            .ok_or_else(|| "line index out of range".to_string())
    }

    fn db_line_mut(&mut self, index: usize) -> Result<&mut Value, String> {
        as_array_mut(get_value_mut(as_object_mut(&mut self.db)?, "lines")?)?
            .get_mut(index)
            .ok_or_else(|| "line index out of range".to_string())
    }

    fn db_bill(&self, index: usize) -> Result<&Value, String> {
        as_array(get_value(as_object(&self.db)?, "bills")?)?
            .get(index)
            .ok_or_else(|| "bill index out of range".to_string())
    }

    fn db_bill_mut(&mut self, index: usize) -> Result<&mut Value, String> {
        as_array_mut(get_value_mut(as_object_mut(&mut self.db)?, "bills")?)?
            .get_mut(index)
            .ok_or_else(|| "bill index out of range".to_string())
    }

    fn find_customer_index_by_id(&self, customer_id: &str) -> Result<usize, String> {
        self.db_customers()?
            .iter()
            .position(|customer| {
                customer
                    .get(&"customer_id")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value == customer_id)
            })
            .ok_or_else(|| format!("Customer with ID {customer_id} not found"))
    }

    fn find_customer_by_id(&self, customer_id: &str) -> Result<&Value, String> {
        let idx = self.find_customer_index_by_id(customer_id)?;
        self.db_customer(idx)
    }

    fn find_customer_index_by_phone_or_line(&self, phone_number: &str) -> Result<usize, String> {
        for (idx, customer) in self.db_customers()?.iter().enumerate() {
            let object = as_object(customer)?;
            if object.get(&"phone_number").and_then(Value::as_str) == Some(phone_number) {
                return Ok(idx);
            }
            for line_id in as_array(get_value(object, "line_ids")?)?
                .iter()
                .filter_map(Value::as_str)
            {
                if self
                    .find_line_by_id(line_id)?
                    .get(&"phone_number")
                    .and_then(Value::as_str)
                    == Some(phone_number)
                {
                    return Ok(idx);
                }
            }
        }
        Err(format!(
            "Customer with phone number {phone_number} not found"
        ))
    }

    fn find_line_index_by_id(&self, line_id: &str) -> Result<usize, String> {
        as_array(get_value(as_object(&self.db)?, "lines")?)?
            .iter()
            .position(|line| line.get(&"line_id").and_then(Value::as_str) == Some(line_id))
            .ok_or_else(|| format!("Line with ID {line_id} not found"))
    }

    fn find_line_by_id(&self, line_id: &str) -> Result<&Value, String> {
        self.db_line(self.find_line_index_by_id(line_id)?)
    }

    fn find_line_index_by_phone(&self, phone_number: &str) -> Result<usize, String> {
        as_array(get_value(as_object(&self.db)?, "lines")?)?
            .iter()
            .position(|line| {
                line.get(&"phone_number").and_then(Value::as_str) == Some(phone_number)
            })
            .ok_or_else(|| format!("Line with phone number {phone_number} not found"))
    }

    fn plan_by_id(&self, plan_id: &str) -> Result<&Value, String> {
        as_array(get_value(as_object(&self.db)?, "plans")?)?
            .iter()
            .find(|plan| plan.get(&"plan_id").and_then(Value::as_str) == Some(plan_id))
            .ok_or_else(|| format!("Plan with ID {plan_id} not found"))
    }

    fn find_device_by_id(&self, device_id: &str) -> Result<&Value, String> {
        as_array(get_value(as_object(&self.db)?, "devices")?)?
            .iter()
            .find(|device| device.get(&"device_id").and_then(Value::as_str) == Some(device_id))
            .ok_or_else(|| format!("Device with ID {device_id} not found"))
    }

    fn find_bill_index_by_id(&self, bill_id: &str) -> Result<usize, String> {
        as_array(get_value(as_object(&self.db)?, "bills")?)?
            .iter()
            .position(|bill| bill.get(&"bill_id").and_then(Value::as_str) == Some(bill_id))
            .ok_or_else(|| format!("Bill with ID {bill_id} not found"))
    }

    fn find_bill_by_id(&self, bill_id: &str) -> Result<&Value, String> {
        self.db_bill(self.find_bill_index_by_id(bill_id)?)
    }

    fn awaiting_payment_bills(&self, customer: &Value) -> Result<Vec<Value>, String> {
        Ok(as_array(get_value(as_object(customer)?, "bill_ids")?)?
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|bill_id| self.find_bill_by_id(bill_id).ok())
            .filter(|bill| bill.get(&"status").and_then(Value::as_str) == Some("Awaiting Payment"))
            .cloned()
            .collect())
    }

    fn first_awaiting_payment_bill(&self, customer: &Value) -> Result<Option<Value>, String> {
        Ok(self.awaiting_payment_bills(customer)?.into_iter().next())
    }

    fn ensure_customer_has_line(&self, customer_id: &str, line_id: &str) -> Result<(), String> {
        let customer = self.find_customer_by_id(customer_id)?;
        let has_line = as_array(get_value(as_object(customer)?, "line_ids")?)?
            .iter()
            .filter_map(Value::as_str)
            .any(|candidate| candidate == line_id);
        if has_line {
            Ok(())
        } else {
            Err(format!(
                "Line {line_id} not found for customer {customer_id}"
            ))
        }
    }

    fn set_bill_to_paid(&mut self, bill_id: &str) -> Result<(), String> {
        let idx = self.find_bill_index_by_id(bill_id)?;
        as_object_mut(self.db_bill_mut(idx)?)?.insert(&"status", json!("Paid".to_string()));
        Ok(())
    }

    fn suspend_line_for_overdue_bill(
        &mut self,
        customer_id: &str,
        line_id: &str,
        new_bill_id: &str,
        contract_ended: bool,
    ) -> Result<Value, String> {
        self.ensure_customer_has_line(customer_id, line_id)?;
        let line_idx = self.find_line_index_by_id(line_id)?;
        if get_string_field(as_object(self.db_line(line_idx)?)?, "status")? != "Active" {
            return Err("Line must be active to suspend for unpaid bill".to_string());
        }
        if self
            .find_customer_by_id(customer_id)?
            .get(&"bill_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .any(|bill_id| {
                self.find_bill_by_id(bill_id)
                    .ok()
                    .and_then(|bill| bill.get(&"status").and_then(Value::as_str))
                    == Some("Overdue")
            })
        {
            return Err("Customer already has an overdue bill".to_string());
        }
        let line = self.db_line(line_idx)?.clone();
        let plan = self
            .plan_by_id(get_string_field(as_object(&line)?, "plan_id")?)?
            .clone();
        let today = parse_date(TODAY)?;
        let first_day_this_month = NaiveDate::from_ymd_opt(today.year(), today.month(), 1).unwrap();
        let last_day_previous_month = first_day_this_month - Duration::days(1);
        let first_day_previous_month = NaiveDate::from_ymd_opt(
            last_day_previous_month.year(),
            last_day_previous_month.month(),
            1,
        )
        .unwrap();
        as_array_mut(get_value_mut(as_object_mut(&mut self.db)?, "bills")?)?.push(json!({
            "bill_id": new_bill_id,
            "customer_id": customer_id,
            "period_start": first_day_previous_month.to_string(),
            "period_end": last_day_previous_month.to_string(),
            "issue_date": first_day_previous_month.to_string(),
            "total_due": get_f64_field(as_object(&plan)?, "price_per_month")?,
            "due_date": (first_day_previous_month + Duration::days(14)).to_string(),
            "status": "Overdue",
            "line_items": [{
                "description": format!("Charge for line {line_id}"),
                "amount": get_f64_field(as_object(&plan)?, "price_per_month")?,
                "date": TODAY,
                "item_type": "Charge",
            }],
        }));
        let customer_idx = self.find_customer_index_by_id(customer_id)?;
        as_array_mut(get_value_mut(
            as_object_mut(self.db_customer_mut(customer_idx)?)?,
            "bill_ids",
        )?)?
        .push(json!(new_bill_id.to_string()));
        let line = self.db_line_mut(line_idx)?;
        as_object_mut(line)?.insert(&"status", json!("Suspended".to_string()));
        as_object_mut(line)?.insert(&"suspension_start_date", json!(TODAY.to_string()));
        if contract_ended {
            as_object_mut(line)?.insert(
                &"contract_end_date",
                json!(last_day_previous_month.to_string()),
            );
        }
        self.sync_tools()?;
        Ok(json!(format!(
            "Line {line_id} suspended for unpaid bill {new_bill_id}. Contract ended: {contract_ended}"
        )))
    }
}

impl TauDomainEnv for TelecomEnv {
    fn policy(&self) -> &str {
        &self.policy
    }

    fn assistant_tools(&self) -> &'static [ToolSpec] {
        TELECOM_ASSISTANT_TOOLS
    }

    fn user_tools(&self) -> &'static [ToolSpec] {
        TELECOM_USER_TOOLS
    }

    fn update_agent_data(&mut self, data: &Value) -> Result<(), String> {
        update_json(&mut self.db, data)?;
        self.sync_tools()
    }

    fn update_user_data(&mut self, data: &Value) -> Result<(), String> {
        update_json(&mut self.user_db, data)?;
        ensure_surroundings(&mut self.user_db)?;
        self.sync_tools()
    }

    fn run_env_function(&mut self, action: &EnvFunctionCall) -> Result<Value, String> {
        match action.env_type {
            ToolRequestor::Assistant => match action.func_name.as_str() {
                "enable_roaming" => self.enable_roaming(
                    str_arg(&action.arguments, "customer_id")?,
                    str_arg(&action.arguments, "line_id")?,
                ),
                "disable_roaming" => self.disable_roaming(
                    str_arg(&action.arguments, "customer_id")?,
                    str_arg(&action.arguments, "line_id")?,
                ),
                "set_data_usage" => {
                    self.ensure_customer_has_line(
                        str_arg(&action.arguments, "customer_id")?,
                        str_arg(&action.arguments, "line_id")?,
                    )?;
                    let idx = self.find_line_index_by_id(str_arg(&action.arguments, "line_id")?)?;
                    as_object_mut(self.db_line_mut(idx)?)?.insert(
                        &"data_used_gb",
                        json!(num_arg(&action.arguments, "data_used_gb")?),
                    );
                    self.sync_tools()?;
                    Ok(json!("Data usage updated".to_string()))
                }
                "suspend_line_for_overdue_bill" => self.suspend_line_for_overdue_bill(
                    str_arg(&action.arguments, "customer_id")?,
                    str_arg(&action.arguments, "line_id")?,
                    str_arg(&action.arguments, "new_bill_id")?,
                    bool_arg(&action.arguments, "contract_ended")?,
                ),
                other => Err(format!("unsupported assistant env function `{other}`")),
            },
            ToolRequestor::User => match action.func_name.as_str() {
                "set_user_info" => {
                    self.surroundings_mut()?.insert(
                        &"name",
                        json!(str_arg(&action.arguments, "name")?.to_string()),
                    );
                    self.surroundings_mut()?.insert(
                        &"phone_number",
                        json!(str_arg(&action.arguments, "phone_number")?.to_string()),
                    );
                    self.sync_tools()?;
                    Ok(json!("User info set".to_string()))
                }
                "set_user_location" => {
                    self.surroundings_mut()?
                        .insert(&"is_abroad", json!(bool_arg(&action.arguments, "abroad")?));
                    self.sync_tools()?;
                    Ok(json!("User location set".to_string()))
                }
                "turn_roaming_off" => {
                    self.device_mut()?.insert(&"roaming_enabled", json!(false));
                    self.simulate_network_search()?;
                    Ok(json!("Data Roaming is now OFF.".to_string()))
                }
                "turn_roaming_on" => {
                    self.device_mut()?.insert(&"roaming_enabled", json!(true));
                    self.simulate_network_search()?;
                    Ok(json!("Data Roaming is now ON.".to_string()))
                }
                "turn_airplane_mode_on" => {
                    self.device_mut()?.insert(&"airplane_mode", json!(true));
                    self.simulate_network_search()?;
                    Ok(json!("Airplane Mode is now ON.".to_string()))
                }
                "turn_data_off" => {
                    self.device_mut()?.insert(&"data_enabled", json!(false));
                    self.simulate_network_search()?;
                    Ok(json!("Data connection broken.".to_string()))
                }
                "turn_data_saver_mode_on" => {
                    self.device_mut()?.insert(&"data_saver_mode", json!(true));
                    Ok(json!("Data Saver Mode is now ON.".to_string()))
                }
                "set_network_mode_preference" => Ok(json!(
                    self.set_network_mode_preference(str_arg(&action.arguments, "mode")?)?
                )),
                "set_wifi_calling" => {
                    self.device_mut()?.insert(
                        &"wifi_calling_enabled",
                        json!(bool_arg(&action.arguments, "enabled")?),
                    );
                    if let Some(value) = action
                        .arguments
                        .get(&"mms_over_wifi")
                        .and_then(Value::as_bool)
                    {
                        self.device_mut()?
                            .insert(&"wifi_calling_mms_over_wifi", json!(value));
                    }
                    Ok(json!("Wi-Fi Calling updated.".to_string()))
                }
                "simulate_network_search" => {
                    self.simulate_network_search()?;
                    Ok(json!("Network search simulated.".to_string()))
                }
                "break_apn_settings" => {
                    self.active_apn_mut()?
                        .as_object_mut()
                        .unwrap()
                        .insert(&"apn_name", json!("broken".to_string()));
                    self.simulate_network_search()?;
                    Ok(json!(
                        "APN settings broken. Please call reset_apn_settings() to fix.".to_string()
                    ))
                }
                "break_apn_mms_setting" => {
                    self.active_apn_mut()?
                        .as_object_mut()
                        .unwrap()
                        .insert(&"mmsc_url", Value::new_null());
                    Ok(json!(
                        "APN MMS setting broken. Please call reset_apn_settings() to fix."
                            .to_string()
                    ))
                }
                "break_vpn" => {
                    self.device_mut()?.insert(&"vpn_connected", json!(true));
                    self.device_mut()?.insert(
                        &"vpn_details",
                        json!({"server_address":"192.168.1.1","protocol":"OpenVPN","server_performance":"poor"}),
                    );
                    Ok(json!("VPN connection broken.".to_string()))
                }
                "lock_sim_card" => {
                    self.device_mut()?.insert(
                        &"sim_card_status",
                        json!(
                            str_arg(&action.arguments, "mode")?
                                .replace("pin", "locked_pin")
                                .replace("puk", "locked_puk")
                        ),
                    );
                    self.simulate_network_search()?;
                    Ok(json!("SIM card locked successfully.".to_string()))
                }
                "remove_app_permission" => {
                    let permission = str_arg(&action.arguments, "permission")?;
                    self.app_status_mut(str_arg(&action.arguments, "app_name")?)?
                        .get_mut(&"permissions")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| "app permissions missing".to_string())?
                        .insert(&permission, json!(false));
                    Ok(json!("App permission removed.".to_string()))
                }
                "unseat_sim_card" => {
                    self.device_mut()?.insert(&"sim_card_missing", json!(true));
                    self.simulate_network_search()?;
                    Ok(json!("SIM card un-seated successfully.".to_string()))
                }
                other => Err(format!("unsupported user env function `{other}`")),
            },
        }
    }

    fn execute_tool_call(&mut self, tool_call: &FunctionCall) -> Result<Value, String> {
        let result = match tool_call.requestor {
            ToolRequestor::Assistant => {
                self.execute_assistant_tool(&tool_call.name, &tool_call.arguments)
            }
            ToolRequestor::User => self.execute_user_tool(&tool_call.name, &tool_call.arguments),
        };
        if result.is_ok() {
            self.sync_tools()?;
        }
        result
    }

    fn run_env_assertion(&self, assertion: &EnvAssertion) -> Result<bool, String> {
        match (assertion.env_type, assertion.func_name.as_str()) {
            (ToolRequestor::Assistant, "assert_data_refueling_amount") => {
                let line = self.find_line_by_id(str_arg(&assertion.arguments, "line_id")?)?;
                Ok((get_f64_field(as_object(line)?, "data_refueling_gb")?
                    - num_arg(&assertion.arguments, "expected_amount")?)
                .abs()
                    < 1e-6)
            }
            (ToolRequestor::Assistant, "assert_no_overdue_bill") => {
                let overdue_bill_id = str_arg(&assertion.arguments, "overdue_bill_id")?;
                match self.find_bill_by_id(overdue_bill_id) {
                    Ok(bill) => Ok(get_string_field(as_object(bill)?, "status")? == "Paid"),
                    Err(_) => Ok(true),
                }
            }
            (ToolRequestor::User, "assert_service_status") => {
                Ok(
                    get_string_field(self.device()?, "network_connection_status")?
                        == str_arg(&assertion.arguments, "expected_status")?,
                )
            }
            (ToolRequestor::User, "assert_mobile_data_status") => {
                Ok(self.mobile_data_working()?
                    == bool_arg(&assertion.arguments, "expected_status")?)
            }
            (ToolRequestor::User, "assert_internet_speed") => {
                let (speed, desc) = self.speed_test();
                Ok(
                    speed.unwrap_or(0.0) >= num_arg(&assertion.arguments, "expected_speed")?
                        && desc
                            .eq_ignore_ascii_case(str_arg(&assertion.arguments, "expected_desc")?),
                )
            }
            (ToolRequestor::User, "assert_can_send_mms") => {
                Ok(self.can_send_mms_flag()? == bool_arg(&assertion.arguments, "expected_status")?)
            }
            _ => Err(format!(
                "unsupported telecom env assertion {}.{}",
                assertion.env_type.as_str(),
                assertion.func_name
            )),
        }
    }

    fn agent_db(&self) -> Option<Value> {
        Some(self.db.clone())
    }

    fn user_db(&self) -> Option<Value> {
        Some(self.user_db.clone())
    }
}

fn parse_toml_value(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value = toml::from_str::<toml::Value>(&text)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    sonic_rs::to_value(&value)
        .map_err(|err| format!("failed to convert {} to json: {err}", path.display()))
}

fn ensure_surroundings(user_db: &mut Value) -> Result<(), String> {
    let object = as_object_mut(user_db)?;
    if !object.contains_key(&"surroundings") {
        object.insert(
            &"surroundings",
            json!({
                "name": null,
                "phone_number": null,
                "is_abroad": false,
                "roaming_allowed": false,
                "signal_strength": {
                    "2G": "poor",
                    "3G": "fair",
                    "4G": "good",
                    "5G": "excellent",
                },
                "mobile_data_usage_exceeded": false,
                "line_active": true,
                "payment_request": null,
            }),
        );
    }
    Ok(())
}

fn str_arg<'a>(args: &'a Map, key: &str) -> Result<&'a str, String> {
    args.get(&key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{key}`"))
}

fn num_arg(args: &Map, key: &str) -> Result<f64, String> {
    args.get(&key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric argument `{key}`"))
}

fn bool_arg(args: &Map, key: &str) -> Result<bool, String> {
    args.get(&key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("missing bool argument `{key}`"))
}

fn yes_no(value: bool) -> &'static str {
    if value { "ON" } else { "OFF" }
}

fn parse_date(value: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .map_err(|err| format!("invalid date {value}: {err}"))
}

fn get_value_mut<'a>(object: &'a mut Map, key: &str) -> Result<&'a mut Value, String> {
    object
        .get_mut(&key)
        .ok_or_else(|| format!("missing field `{key}`"))
}
