use super::tau_bench_common::{
    EnvAssertion, EnvFunctionCall, TauDomainEnv, ToolArgSpec, ToolSpec, as_array, as_array_mut,
    as_object, as_object_mut, calculate_expression, get_f64_field, get_string_field, get_value,
    update_json,
};
use super::{FunctionCall, ToolRequestor};
use sonic_rs::{Object as Map, Value, json, prelude::*};
use std::fs;
use std::path::Path;

const EMPTY_ARGS: &[ToolArgSpec] = &[];
const EMAIL_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "email",
    description: "customer email address",
}];
const USER_ID_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "user_id",
    description: "retail user id",
}];
const ORDER_ID_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "order_id",
    description: "order id such as #W0000000",
}];
const PRODUCT_ID_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "product_id",
    description: "product id, not item id",
}];
const CALCULATE_ARGS: &[ToolArgSpec] = &[ToolArgSpec {
    name: "expression",
    description: "math expression using digits and +-*/()",
}];
const NAME_ZIP_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "first_name",
        description: "customer first name",
    },
    ToolArgSpec {
        name: "last_name",
        description: "customer last name",
    },
    ToolArgSpec {
        name: "zip",
        description: "postal code",
    },
];
const CANCEL_ORDER_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "order_id",
        description: "pending order id",
    },
    ToolArgSpec {
        name: "reason",
        description: "either `no longer needed` or `ordered by mistake`",
    },
];
const ITEM_EXCHANGE_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "order_id",
        description: "order id",
    },
    ToolArgSpec {
        name: "item_ids",
        description: "array of existing item ids, duplicates allowed",
    },
    ToolArgSpec {
        name: "new_item_ids",
        description: "array of replacement item ids aligned with item_ids",
    },
    ToolArgSpec {
        name: "payment_method_id",
        description: "payment method used to pay or receive any price difference",
    },
];
const RETURN_ITEMS_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "order_id",
        description: "delivered order id",
    },
    ToolArgSpec {
        name: "item_ids",
        description: "array of item ids to return, duplicates allowed",
    },
    ToolArgSpec {
        name: "payment_method_id",
        description: "refund payment method id, either original method or a gift card",
    },
];
const PAYMENT_METHOD_ARG: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "order_id",
        description: "pending order id",
    },
    ToolArgSpec {
        name: "payment_method_id",
        description: "new payment method id",
    },
];
const ADDRESS_ARGS_WITH_ORDER: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "order_id",
        description: "pending order id",
    },
    ToolArgSpec {
        name: "address1",
        description: "address line 1",
    },
    ToolArgSpec {
        name: "address2",
        description: "address line 2 or empty string",
    },
    ToolArgSpec {
        name: "city",
        description: "city",
    },
    ToolArgSpec {
        name: "state",
        description: "state or province",
    },
    ToolArgSpec {
        name: "country",
        description: "country",
    },
    ToolArgSpec {
        name: "zip",
        description: "postal code",
    },
];
const ADDRESS_ARGS_WITH_USER: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "user_id",
        description: "retail user id",
    },
    ToolArgSpec {
        name: "address1",
        description: "address line 1",
    },
    ToolArgSpec {
        name: "address2",
        description: "address line 2 or empty string",
    },
    ToolArgSpec {
        name: "city",
        description: "city",
    },
    ToolArgSpec {
        name: "state",
        description: "state or province",
    },
    ToolArgSpec {
        name: "country",
        description: "country",
    },
    ToolArgSpec {
        name: "zip",
        description: "postal code",
    },
];
const SUMMARY_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "summary",
    description: "brief handoff summary",
}];

const RETAIL_ASSISTANT_TOOLS: &[ToolSpec] = &[
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "calculate",
        description: "Evaluate a simple arithmetic expression.",
        arguments: CALCULATE_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "cancel_pending_order",
        description: "Cancel a pending order and append refunds to its payment history.",
        arguments: CANCEL_ORDER_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "exchange_delivered_order_items",
        description: "Request an exchange for delivered order items with same-product replacements.",
        arguments: ITEM_EXCHANGE_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "find_user_id_by_name_zip",
        description: "Authenticate a retail user by first name, last name, and zip code.",
        arguments: NAME_ZIP_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "find_user_id_by_email",
        description: "Authenticate a retail user by email address.",
        arguments: EMAIL_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_order_details",
        description: "Read order details and current order status.",
        arguments: ORDER_ID_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_product_details",
        description: "Read product details and all variant items for a product type.",
        arguments: PRODUCT_ID_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_user_details",
        description: "Read a user's profile, default address, payment methods, and orders.",
        arguments: USER_ID_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "list_all_product_types",
        description: "List all product types as a mapping from product name to product id.",
        arguments: EMPTY_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "modify_pending_order_address",
        description: "Modify the shipping address on a pending order.",
        arguments: ADDRESS_ARGS_WITH_ORDER,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "modify_pending_order_items",
        description: "Modify pending order items to same-product replacement variants.",
        arguments: ITEM_EXCHANGE_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "modify_pending_order_payment",
        description: "Change the payment method on a pending order.",
        arguments: PAYMENT_METHOD_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "modify_user_address",
        description: "Modify a user's default address.",
        arguments: ADDRESS_ARGS_WITH_USER,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "return_delivered_order_items",
        description: "Request a return for delivered order items.",
        arguments: RETURN_ITEMS_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "transfer_to_human_agents",
        description: "Transfer the case to a human agent.",
        arguments: SUMMARY_ARG,
    },
];

pub struct RetailEnv {
    policy: String,
    db: Value,
}

impl RetailEnv {
    pub fn load(dataset_root: &Path) -> Result<Self, String> {
        let base = dataset_root.join("tau_bench").join("retail");
        let db = sonic_rs::from_str::<Value>(
            &fs::read_to_string(base.join("db.json")).map_err(|err| err.to_string())?,
        )
        .map_err(|err| format!("failed to parse retail db.json: {err}"))?;
        let policy = fs::read_to_string(base.join("policy.md")).map_err(|err| err.to_string())?;
        Ok(Self { policy, db })
    }

    fn products(&self) -> Result<&Map, String> {
        let root = as_object(&self.db)?;
        let products = get_value(root, "products")?;
        as_object(products)
    }

    fn users(&self) -> Result<&Map, String> {
        let root = as_object(&self.db)?;
        let users = get_value(root, "users")?;
        as_object(users)
    }

    fn orders(&self) -> Result<&Map, String> {
        let root = as_object(&self.db)?;
        let orders = get_value(root, "orders")?;
        as_object(orders)
    }

    fn users_mut(&mut self) -> Result<&mut Map, String> {
        let root = as_object_mut(&mut self.db)?;
        let users = root
            .get_mut(&"users")
            .ok_or_else(|| "missing users in retail db".to_string())?;
        as_object_mut(users)
    }

    fn orders_mut(&mut self) -> Result<&mut Map, String> {
        let root = as_object_mut(&mut self.db)?;
        let orders = root
            .get_mut(&"orders")
            .ok_or_else(|| "missing orders in retail db".to_string())?;
        as_object_mut(orders)
    }

    fn get_user(&self, user_id: &str) -> Result<&Value, String> {
        self.users()?
            .get(&user_id)
            .ok_or_else(|| "User not found".to_string())
    }

    fn get_user_clone(&self, user_id: &str) -> Result<Value, String> {
        Ok(self.get_user(user_id)?.clone())
    }

    fn set_user(&mut self, user_id: &str, user: Value) -> Result<(), String> {
        self.users_mut()?.insert(&user_id, user);
        Ok(())
    }

    fn get_order(&self, order_id: &str) -> Result<&Value, String> {
        self.orders()?
            .get(&order_id)
            .ok_or_else(|| "Order not found".to_string())
    }

    fn get_order_clone(&self, order_id: &str) -> Result<Value, String> {
        Ok(self.get_order(order_id)?.clone())
    }

    fn set_order(&mut self, order_id: &str, order: Value) -> Result<(), String> {
        self.orders_mut()?.insert(&order_id, order);
        Ok(())
    }

    fn get_product(&self, product_id: &str) -> Result<&Value, String> {
        self.products()?
            .get(&product_id)
            .ok_or_else(|| "Product not found".to_string())
    }

    fn get_product_clone(&self, product_id: &str) -> Result<Value, String> {
        Ok(self.get_product(product_id)?.clone())
    }

    fn get_variant(&self, product_id: &str, item_id: &str) -> Result<&Value, String> {
        let product = self.get_product(product_id)?;
        let variants = as_object(
            as_object(product)?
                .get(&"variants")
                .ok_or_else(|| "missing variants".to_string())?,
        )?;
        variants
            .get(&item_id)
            .ok_or_else(|| "Variant not found".to_string())
    }

    fn is_pending_order(order: &Value) -> Result<bool, String> {
        Ok(get_string_field(as_object(order)?, "status")?.contains("pending"))
    }

    fn retail_payment_methods<'a>(user: &'a Value) -> Result<&'a Map, String> {
        let user_object = as_object(user)?;
        let payment_methods = get_value(user_object, "payment_methods")?;
        as_object(payment_methods)
    }

    fn retail_payment_methods_mut<'a>(user: &'a mut Value) -> Result<&'a mut Map, String> {
        let user_object = as_object_mut(user)?;
        let payment_methods = user_object
            .get_mut(&"payment_methods")
            .ok_or_else(|| "missing payment_methods".to_string())?;
        as_object_mut(payment_methods)
    }

    fn get_payment_method_source(user: &Value, payment_method_id: &str) -> Result<String, String> {
        let payment_methods = Self::retail_payment_methods(user)?;
        let payment_method = payment_methods
            .get(&payment_method_id)
            .ok_or_else(|| "Payment method not found".to_string())?;
        Ok(get_string_field(as_object(payment_method)?, "source")?.to_string())
    }

    fn adjust_gift_card_balance(
        user: &mut Value,
        payment_method_id: &str,
        delta: f64,
    ) -> Result<bool, String> {
        let payment_methods = Self::retail_payment_methods_mut(user)?;
        let payment_method = payment_methods
            .get_mut(&payment_method_id)
            .ok_or_else(|| "Payment method not found".to_string())?;
        let payment_method_object = as_object_mut(payment_method)?;
        if get_string_field(payment_method_object, "source")? != "gift_card" {
            return Ok(false);
        }
        let balance = get_f64_field(payment_method_object, "balance")?;
        payment_method_object.insert(&"balance", json!(round_money(balance + delta)));
        Ok(true)
    }

    fn ensure_payment_method_exists(user: &Value, payment_method_id: &str) -> Result<(), String> {
        let payment_methods = Self::retail_payment_methods(user)?;
        if payment_methods.contains_key(&payment_method_id) {
            Ok(())
        } else {
            Err("Payment method not found".to_string())
        }
    }

    fn build_address(args: &Map) -> Result<Value, String> {
        Ok(json!({
            "address1": get_arg_str(args, "address1")?,
            "address2": get_arg_str(args, "address2")?,
            "city": get_arg_str(args, "city")?,
            "state": get_arg_str(args, "state")?,
            "country": get_arg_str(args, "country")?,
            "zip": get_arg_str(args, "zip")?,
        }))
    }

    fn execute_retail_tool(&mut self, tool_call: &FunctionCall) -> Result<Value, String> {
        let args = &tool_call.arguments;
        match tool_call.name.as_str() {
            "calculate" => Ok(json!(calculate_expression(get_arg_str(
                args,
                "expression",
            )?)?)),
            "cancel_pending_order" => self
                .cancel_pending_order(get_arg_str(args, "order_id")?, get_arg_str(args, "reason")?),
            "exchange_delivered_order_items" => self.exchange_delivered_order_items(args),
            "find_user_id_by_name_zip" => self.find_user_id_by_name_zip(
                get_arg_str(args, "first_name")?,
                get_arg_str(args, "last_name")?,
                get_arg_str(args, "zip")?,
            ),
            "find_user_id_by_email" => self.find_user_id_by_email(get_arg_str(args, "email")?),
            "get_order_details" => Ok(self.get_order_clone(get_arg_str(args, "order_id")?)?),
            "get_product_details" => Ok(self.get_product_clone(get_arg_str(args, "product_id")?)?),
            "get_user_details" => Ok(self.get_user_clone(get_arg_str(args, "user_id")?)?),
            "list_all_product_types" => self.list_all_product_types(),
            "modify_pending_order_address" => self.modify_pending_order_address(args),
            "modify_pending_order_items" => self.modify_pending_order_items(args),
            "modify_pending_order_payment" => self.modify_pending_order_payment(args),
            "modify_user_address" => self.modify_user_address(args),
            "return_delivered_order_items" => self.return_delivered_order_items(args),
            "transfer_to_human_agents" => Ok(json!("Transfer successful".to_string())),
            other => Err(format!("unsupported retail tool `{other}`")),
        }
    }

    fn find_user_id_by_name_zip(
        &self,
        first_name: &str,
        last_name: &str,
        zip: &str,
    ) -> Result<Value, String> {
        for (user_id, user) in self.users()? {
            let user_object = as_object(user)?;
            let name = as_object(
                user_object
                    .get(&"name")
                    .ok_or_else(|| "missing user name".to_string())?,
            )?;
            let address = as_object(
                user_object
                    .get(&"address")
                    .ok_or_else(|| "missing user address".to_string())?,
            )?;
            if get_string_field(name, "first_name")?.eq_ignore_ascii_case(first_name)
                && get_string_field(name, "last_name")?.eq_ignore_ascii_case(last_name)
                && get_string_field(address, "zip")? == zip
            {
                return Ok(json!(user_id));
            }
        }
        Err("User not found".to_string())
    }

    fn find_user_id_by_email(&self, email: &str) -> Result<Value, String> {
        for (user_id, user) in self.users()? {
            let user_object = as_object(user)?;
            if get_string_field(user_object, "email")?.eq_ignore_ascii_case(email) {
                return Ok(json!(user_id));
            }
        }
        Err("User not found".to_string())
    }

    fn list_all_product_types(&self) -> Result<Value, String> {
        let mut product_types = Map::new();
        for (_, product) in self.products()?.iter() {
            let product_object = as_object(product)?;
            product_types.insert(
                get_string_field(product_object, "name")?,
                json!(get_string_field(product_object, "product_id")?.to_string()),
            );
        }
        Ok(json!(product_types))
    }

    fn cancel_pending_order(&mut self, order_id: &str, reason: &str) -> Result<Value, String> {
        let mut order = self.get_order_clone(order_id)?;
        if get_string_field(as_object(&order)?, "status")? != "pending" {
            return Err("Non-pending order cannot be cancelled".to_string());
        }
        if !matches!(reason, "no longer needed" | "ordered by mistake") {
            return Err("Invalid reason".to_string());
        }

        let user_id = get_string_field(as_object(&order)?, "user_id")?.to_string();
        let mut user = self.get_user_clone(&user_id)?;
        let payments = as_array(
            as_object(&order)?
                .get(&"payment_history")
                .ok_or_else(|| "missing payment_history".to_string())?,
        )?
        .clone();

        let mut refunds = Vec::new();
        for payment in payments {
            let payment_object = as_object(&payment)?;
            let payment_method_id = get_string_field(payment_object, "payment_method_id")?;
            let amount = get_f64_field(payment_object, "amount")?;
            refunds.push(json!({
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": payment_method_id,
            }));
            let _ = Self::adjust_gift_card_balance(&mut user, payment_method_id, amount)?;
        }

        let order_object = as_object_mut(&mut order)?;
        as_array_mut(
            order_object
                .get_mut(&"payment_history")
                .ok_or_else(|| "missing payment_history".to_string())?,
        )?
        .extend(refunds.iter());
        order_object.insert(&"status", json!("cancelled".to_string()));
        order_object.insert(&"cancel_reason", json!(reason.to_string()));

        self.set_user(&user_id, user)?;
        self.set_order(order_id, order.clone())?;
        Ok(order)
    }

    fn exchange_delivered_order_items(&mut self, args: &Map) -> Result<Value, String> {
        let order_id = get_arg_str(args, "order_id")?;
        let item_ids = get_arg_string_vec(args, "item_ids")?;
        let new_item_ids = get_arg_string_vec(args, "new_item_ids")?;
        let payment_method_id = get_arg_str(args, "payment_method_id")?;

        let mut order = self.get_order_clone(order_id)?;
        if get_string_field(as_object(&order)?, "status")? != "delivered" {
            return Err("Non-delivered order cannot be exchanged".to_string());
        }

        let all_item_ids = order_item_ids(&order)?;
        for item_id in &item_ids {
            if count_str(&item_ids, item_id) > count_str(&all_item_ids, item_id) {
                return Err(format!("Number of {item_id} not found."));
            }
        }
        if item_ids.len() != new_item_ids.len() {
            return Err("The number of items to be exchanged should match.".to_string());
        }

        let mut diff_price = 0.0_f64;
        for (item_id, new_item_id) in item_ids.iter().zip(&new_item_ids) {
            let item = find_order_item(&order, item_id)?
                .ok_or_else(|| format!("Item {item_id} not found"))?;
            let item_object = as_object(item)?;
            let product_id = get_string_field(item_object, "product_id")?;
            let variant = self.get_variant(product_id, new_item_id)?;
            let variant_object = as_object(variant)?;
            if !variant_object
                .get(&"available")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                return Err(format!("New item {new_item_id} not found or available"));
            }
            diff_price +=
                get_f64_field(variant_object, "price")? - get_f64_field(item_object, "price")?;
        }
        diff_price = round_money(diff_price);

        let user_id = get_string_field(as_object(&order)?, "user_id")?.to_string();
        let user = self.get_user_clone(&user_id)?;
        Self::ensure_payment_method_exists(&user, payment_method_id)?;
        if Self::get_payment_method_source(&user, payment_method_id)? == "gift_card" {
            let payment_methods = Self::retail_payment_methods(&user)?;
            let payment_method = payment_methods
                .get(&payment_method_id)
                .ok_or_else(|| "Payment method not found".to_string())?;
            if get_f64_field(as_object(payment_method)?, "balance")? < diff_price {
                return Err(
                    "Insufficient gift card balance to pay for the price difference".to_string(),
                );
            }
        }

        let order_object = as_object_mut(&mut order)?;
        order_object.insert(&"status", json!("exchange requested".to_string()));
        order_object.insert(&"exchange_items", json!(sorted_strings(item_ids)));
        order_object.insert(&"exchange_new_items", json!(sorted_strings(new_item_ids)));
        order_object.insert(
            &"exchange_payment_method_id",
            json!(payment_method_id.to_string()),
        );
        order_object.insert(&"exchange_price_difference", json!(diff_price));

        self.set_order(order_id, order.clone())?;
        Ok(order)
    }

    fn modify_pending_order_address(&mut self, args: &Map) -> Result<Value, String> {
        let order_id = get_arg_str(args, "order_id")?;
        let mut order = self.get_order_clone(order_id)?;
        if !Self::is_pending_order(&order)? {
            return Err("Non-pending order cannot be modified".to_string());
        }
        as_object_mut(&mut order)?.insert(&"address", Self::build_address(args)?);
        self.set_order(order_id, order.clone())?;
        Ok(order)
    }

    fn modify_pending_order_items(&mut self, args: &Map) -> Result<Value, String> {
        let order_id = get_arg_str(args, "order_id")?;
        let item_ids = get_arg_string_vec(args, "item_ids")?;
        let new_item_ids = get_arg_string_vec(args, "new_item_ids")?;
        let payment_method_id = get_arg_str(args, "payment_method_id")?;

        let mut order = self.get_order_clone(order_id)?;
        if get_string_field(as_object(&order)?, "status")? != "pending" {
            return Err("Non-pending order cannot be modified".to_string());
        }

        let all_item_ids = order_item_ids(&order)?;
        for item_id in &item_ids {
            if count_str(&item_ids, item_id) > count_str(&all_item_ids, item_id) {
                return Err(format!("{item_id} not found"));
            }
        }
        if item_ids.len() != new_item_ids.len() {
            return Err("The number of items to be exchanged should match".to_string());
        }

        let mut diff_price = 0.0_f64;
        let mut replacements = Vec::with_capacity(item_ids.len());
        for (item_id, new_item_id) in item_ids.iter().zip(&new_item_ids) {
            if item_id == new_item_id {
                return Err("The new item id should be different from the old item id".to_string());
            }
            let item = find_order_item(&order, item_id)?
                .ok_or_else(|| format!("Item {item_id} not found"))?;
            let item_object = as_object(item)?;
            let product_id = get_string_field(item_object, "product_id")?.to_string();
            let variant = self.get_variant(&product_id, new_item_id)?;
            let variant_object = as_object(variant)?;
            if !variant_object
                .get(&"available")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                return Err(format!("New item {new_item_id} not found or available"));
            }
            diff_price +=
                get_f64_field(variant_object, "price")? - get_f64_field(item_object, "price")?;
            replacements.push((
                item_id.clone(),
                new_item_id.clone(),
                get_f64_field(variant_object, "price")?,
                get_value(variant_object, "options")?.clone(),
            ));
        }
        diff_price = round_money(diff_price);

        let user_id = get_string_field(as_object(&order)?, "user_id")?.to_string();
        let mut user = self.get_user_clone(&user_id)?;
        Self::ensure_payment_method_exists(&user, payment_method_id)?;
        if Self::get_payment_method_source(&user, payment_method_id)? == "gift_card" {
            let payment_methods = Self::retail_payment_methods(&user)?;
            let payment_method = payment_methods
                .get(&payment_method_id)
                .ok_or_else(|| "Payment method not found".to_string())?;
            if get_f64_field(as_object(payment_method)?, "balance")? < diff_price {
                return Err("Insufficient gift card balance to pay for the new item".to_string());
            }
        }

        {
            let order_object = as_object_mut(&mut order)?;
            as_array_mut(
                order_object
                    .get_mut(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?
            .push(json!({
                "transaction_type": if diff_price > 0.0 { "payment" } else { "refund" },
                "amount": round_money(diff_price.abs()),
                "payment_method_id": payment_method_id,
            }));
        }
        let _ = Self::adjust_gift_card_balance(&mut user, payment_method_id, -diff_price)?;

        {
            let order_object = as_object_mut(&mut order)?;
            let items = as_array_mut(
                order_object
                    .get_mut(&"items")
                    .ok_or_else(|| "missing items".to_string())?,
            )?;
            for (old_item_id, new_item_id, new_price, new_options) in replacements {
                let mut found = false;
                for item in items.iter_mut() {
                    let item_object = as_object_mut(item)?;
                    if get_string_field(item_object, "item_id")? == old_item_id {
                        item_object.insert(&"item_id", json!(new_item_id));
                        item_object.insert(&"price", json!(new_price));
                        item_object.insert(&"options", new_options);
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(format!("Item {old_item_id} not found"));
                }
            }
            order_object.insert(&"status", json!("pending (item modified)".to_string()));
        }

        self.set_user(&user_id, user)?;
        self.set_order(order_id, order.clone())?;
        Ok(order)
    }

    fn modify_pending_order_payment(&mut self, args: &Map) -> Result<Value, String> {
        let order_id = get_arg_str(args, "order_id")?;
        let payment_method_id = get_arg_str(args, "payment_method_id")?;

        let mut order = self.get_order_clone(order_id)?;
        if !Self::is_pending_order(&order)? {
            return Err("Non-pending order cannot be modified".to_string());
        }

        let user_id = get_string_field(as_object(&order)?, "user_id")?.to_string();
        let mut user = self.get_user_clone(&user_id)?;
        Self::ensure_payment_method_exists(&user, payment_method_id)?;

        let first_payment = {
            let order_object = as_object(&order)?;
            let payment_history = as_array(
                order_object
                    .get(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?;
            if payment_history.len() != 1
                || get_string_field(as_object(&payment_history[0])?, "transaction_type")?
                    != "payment"
            {
                return Err("There should be exactly one payment for a pending order".to_string());
            }
            payment_history[0].clone()
        };

        let first_payment_object = as_object(&first_payment)?;
        let original_payment_method_id =
            get_string_field(first_payment_object, "payment_method_id")?.to_string();
        if original_payment_method_id == payment_method_id {
            return Err(
                "The new payment method should be different from the current one".to_string(),
            );
        }
        let amount = get_f64_field(first_payment_object, "amount")?;

        if Self::get_payment_method_source(&user, payment_method_id)? == "gift_card" {
            let payment_methods = Self::retail_payment_methods(&user)?;
            let payment_method = payment_methods
                .get(&payment_method_id)
                .ok_or_else(|| "Payment method not found".to_string())?;
            if get_f64_field(as_object(payment_method)?, "balance")? < amount {
                return Err("Insufficient gift card balance to pay for the order".to_string());
            }
        }

        {
            let order_object = as_object_mut(&mut order)?;
            as_array_mut(
                order_object
                    .get_mut(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?
            .extend(
                [
                    json!({
                        "transaction_type": "payment",
                        "amount": amount,
                        "payment_method_id": payment_method_id,
                    }),
                    json!({
                        "transaction_type": "refund",
                        "amount": amount,
                        "payment_method_id": original_payment_method_id,
                    }),
                ]
                .iter(),
            );
        }

        let _ = Self::adjust_gift_card_balance(&mut user, payment_method_id, -amount)?;
        let _ = Self::adjust_gift_card_balance(&mut user, &original_payment_method_id, amount)?;

        self.set_user(&user_id, user)?;
        self.set_order(order_id, order.clone())?;
        Ok(order)
    }

    fn modify_user_address(&mut self, args: &Map) -> Result<Value, String> {
        let user_id = get_arg_str(args, "user_id")?;
        let mut user = self.get_user_clone(user_id)?;
        as_object_mut(&mut user)?.insert(&"address", Self::build_address(args)?);
        self.set_user(user_id, user.clone())?;
        Ok(user)
    }

    fn return_delivered_order_items(&mut self, args: &Map) -> Result<Value, String> {
        let order_id = get_arg_str(args, "order_id")?;
        let item_ids = get_arg_string_vec(args, "item_ids")?;
        let payment_method_id = get_arg_str(args, "payment_method_id")?;

        let mut order = self.get_order_clone(order_id)?;
        if get_string_field(as_object(&order)?, "status")? != "delivered" {
            return Err("Non-delivered order cannot be returned".to_string());
        }

        let user_id = get_string_field(as_object(&order)?, "user_id")?.to_string();
        let user = self.get_user_clone(&user_id)?;
        Self::ensure_payment_method_exists(&user, payment_method_id)?;
        let payment_source = Self::get_payment_method_source(&user, payment_method_id)?;
        let original_payment_method_id = {
            let order_object = as_object(&order)?;
            let payment_history = as_array(
                order_object
                    .get(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?;
            let first_payment = payment_history
                .first()
                .ok_or_else(|| "missing payment history entry".to_string())?;
            get_string_field(as_object(first_payment)?, "payment_method_id")?.to_string()
        };
        if payment_source != "gift_card" && payment_method_id != original_payment_method_id {
            return Err("Payment method should be the original payment method".to_string());
        }

        let all_item_ids = order_item_ids(&order)?;
        for item_id in &item_ids {
            if count_str(&item_ids, item_id) > count_str(&all_item_ids, item_id) {
                return Err("Some item not found".to_string());
            }
        }

        let order_object = as_object_mut(&mut order)?;
        order_object.insert(&"status", json!("return requested".to_string()));
        order_object.insert(&"return_items", json!(sorted_strings(item_ids)));
        order_object.insert(
            &"return_payment_method_id",
            json!(payment_method_id.to_string()),
        );

        self.set_order(order_id, order.clone())?;
        Ok(order)
    }
}

impl TauDomainEnv for RetailEnv {
    fn policy(&self) -> &str {
        &self.policy
    }

    fn assistant_tools(&self) -> &'static [ToolSpec] {
        RETAIL_ASSISTANT_TOOLS
    }

    fn user_tools(&self) -> &'static [ToolSpec] {
        &[]
    }

    fn update_agent_data(&mut self, data: &Value) -> Result<(), String> {
        update_json(&mut self.db, data)
    }

    fn update_user_data(&mut self, _data: &Value) -> Result<(), String> {
        Err("retail has no separate user db".to_string())
    }

    fn run_env_function(&mut self, action: &EnvFunctionCall) -> Result<Value, String> {
        if action.env_type != ToolRequestor::Assistant {
            return Err("retail only supports assistant env functions".to_string());
        }
        let call = FunctionCall {
            requestor: ToolRequestor::Assistant,
            name: action.func_name.clone(),
            arguments: action.arguments.clone(),
        };
        self.execute_tool_call(&call)
    }

    fn execute_tool_call(&mut self, tool_call: &FunctionCall) -> Result<Value, String> {
        if tool_call.requestor != ToolRequestor::Assistant {
            return Err("retail does not support user tool calls".to_string());
        }
        self.execute_retail_tool(tool_call)
    }

    fn run_env_assertion(&self, _assertion: &EnvAssertion) -> Result<bool, String> {
        Err("retail has no env assertions".to_string())
    }

    fn agent_db(&self) -> Option<Value> {
        Some(self.db.clone())
    }

    fn user_db(&self) -> Option<Value> {
        None
    }
}

fn get_arg_str<'a>(args: &'a Map, key: &str) -> Result<&'a str, String> {
    args.get(&key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string argument `{key}`"))
}

fn get_arg_array<'a>(args: &'a Map, key: &str) -> Result<&'a [Value], String> {
    args.get(&key)
        .and_then(Value::as_array)
        .map(|value| value.as_slice())
        .ok_or_else(|| format!("missing array argument `{key}`"))
}

fn get_arg_string_vec(args: &Map, key: &str) -> Result<Vec<String>, String> {
    get_arg_array(args, key)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| format!("expected all values in `{key}` to be strings"))
        })
        .collect()
}

fn round_money(amount: f64) -> f64 {
    (amount * 100.0).round() / 100.0
}

fn count_str(values: &[String], target: &str) -> usize {
    values
        .iter()
        .filter(|value| value.as_str() == target)
        .count()
}

fn sorted_strings(values: Vec<String>) -> Vec<Value> {
    let mut values = values;
    values.sort();
    values.into_iter().map(|value| json!(value)).collect()
}

fn order_item_ids(order: &Value) -> Result<Vec<String>, String> {
    let order_object = as_object(order)?;
    let items = as_array(
        order_object
            .get(&"items")
            .ok_or_else(|| "missing items".to_string())?,
    )?;
    items
        .iter()
        .map(|item| Ok(get_string_field(as_object(item)?, "item_id")?.to_string()))
        .collect()
}

fn find_order_item<'a>(order: &'a Value, item_id: &str) -> Result<Option<&'a Value>, String> {
    let order_object = as_object(order)?;
    let items = as_array(
        order_object
            .get(&"items")
            .ok_or_else(|| "missing items".to_string())?,
    )?;
    Ok(items.iter().find(|item| {
        as_object(item)
            .ok()
            .and_then(|item_object| item_object.get(&"item_id").and_then(Value::as_str))
            == Some(item_id)
    }))
}
