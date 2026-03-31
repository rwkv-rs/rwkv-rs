use std::path::Path;

use sonic_rs::{Object as Map, Value, json, prelude::*};

use super::{
    EnvAssertion, EnvFunctionCall, FunctionCall, TauDomainEnv, ToolArgSpec, ToolRequestor,
    ToolSpec, as_array, as_array_mut, as_object, as_object_mut, calculate_expression,
    get_f64_field, get_string_field, get_value, update_json,
};

const EMPTY_ARGS: &[ToolArgSpec] = &[];
const USER_ID_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "user_id",
    description: "customer user id",
}];
const RESERVATION_ID_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "reservation_id",
    description: "reservation id",
}];
const SEARCH_DIRECT_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "origin",
        description: "three-letter origin airport",
    },
    ToolArgSpec {
        name: "destination",
        description: "three-letter destination airport",
    },
    ToolArgSpec {
        name: "date",
        description: "flight date in YYYY-MM-DD",
    },
];
const SEARCH_ONESTOP_ARGS: &[ToolArgSpec] = SEARCH_DIRECT_ARGS;
const FLIGHT_STATUS_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "flight_number",
        description: "flight number",
    },
    ToolArgSpec {
        name: "date",
        description: "flight date in YYYY-MM-DD",
    },
];
const SEND_CERTIFICATE_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "user_id",
        description: "customer user id",
    },
    ToolArgSpec {
        name: "amount",
        description: "certificate amount",
    },
];
const SUMMARY_ARG: &[ToolArgSpec] = &[ToolArgSpec {
    name: "summary",
    description: "handoff summary",
}];
const CALCULATE_ARGS: &[ToolArgSpec] = &[ToolArgSpec {
    name: "expression",
    description: "math expression using digits and +-*/()",
}];
const BOOK_RESERVATION_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "user_id",
        description: "customer user id",
    },
    ToolArgSpec {
        name: "origin",
        description: "origin airport code",
    },
    ToolArgSpec {
        name: "destination",
        description: "destination airport code",
    },
    ToolArgSpec {
        name: "flight_type",
        description: "one_way or round_trip",
    },
    ToolArgSpec {
        name: "cabin",
        description: "basic_economy, economy, or business",
    },
    ToolArgSpec {
        name: "flights",
        description: "array of flight_number/date objects",
    },
    ToolArgSpec {
        name: "passengers",
        description: "array of passenger objects",
    },
    ToolArgSpec {
        name: "payment_methods",
        description: "array of payment_id/amount objects",
    },
    ToolArgSpec {
        name: "total_baggages",
        description: "total bag count",
    },
    ToolArgSpec {
        name: "nonfree_baggages",
        description: "paid bag count",
    },
    ToolArgSpec {
        name: "insurance",
        description: "yes or no",
    },
];
const UPDATE_BAGGAGES_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "reservation_id",
        description: "reservation id",
    },
    ToolArgSpec {
        name: "total_baggages",
        description: "new total bag count",
    },
    ToolArgSpec {
        name: "nonfree_baggages",
        description: "new paid bag count",
    },
    ToolArgSpec {
        name: "payment_id",
        description: "payment method id for extra charges",
    },
];
const UPDATE_FLIGHTS_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "reservation_id",
        description: "reservation id",
    },
    ToolArgSpec {
        name: "cabin",
        description: "new cabin class",
    },
    ToolArgSpec {
        name: "flights",
        description: "entire replacement flight list",
    },
    ToolArgSpec {
        name: "payment_id",
        description: "payment method id for price difference",
    },
];
const UPDATE_PASSENGERS_ARGS: &[ToolArgSpec] = &[
    ToolArgSpec {
        name: "reservation_id",
        description: "reservation id",
    },
    ToolArgSpec {
        name: "passengers",
        description: "replacement passenger list",
    },
];

const AIRPORTS: &[(&str, &str)] = &[
    ("SFO", "San Francisco"),
    ("JFK", "New York"),
    ("LAX", "Los Angeles"),
    ("ORD", "Chicago"),
    ("DFW", "Dallas"),
    ("DEN", "Denver"),
    ("SEA", "Seattle"),
    ("ATL", "Atlanta"),
    ("MIA", "Miami"),
    ("BOS", "Boston"),
    ("PHX", "Phoenix"),
    ("IAH", "Houston"),
    ("LAS", "Las Vegas"),
    ("MCO", "Orlando"),
    ("EWR", "Newark"),
    ("CLT", "Charlotte"),
    ("MSP", "Minneapolis"),
    ("DTW", "Detroit"),
    ("PHL", "Philadelphia"),
    ("LGA", "LaGuardia"),
];

const AIRLINE_ASSISTANT_TOOLS: &[ToolSpec] = &[
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "book_reservation",
        description: "Book a reservation and charge the provided payment methods.",
        arguments: BOOK_RESERVATION_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "calculate",
        description: "Evaluate a simple arithmetic expression.",
        arguments: CALCULATE_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "cancel_reservation",
        description: "Cancel an existing reservation and append refund records.",
        arguments: RESERVATION_ID_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_reservation_details",
        description: "Read reservation details by reservation id.",
        arguments: RESERVATION_ID_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_user_details",
        description: "Read user details including saved reservations and payment methods.",
        arguments: USER_ID_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "list_all_airports",
        description: "List all supported airports.",
        arguments: EMPTY_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "search_direct_flight",
        description: "Search available direct flights for a date and route.",
        arguments: SEARCH_DIRECT_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "search_onestop_flight",
        description: "Search available one-stop flights for a date and route.",
        arguments: SEARCH_ONESTOP_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "send_certificate",
        description: "Issue a compensation certificate to the user profile.",
        arguments: SEND_CERTIFICATE_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "transfer_to_human_agents",
        description: "Transfer the case to a human agent.",
        arguments: SUMMARY_ARG,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "update_reservation_baggages",
        description: "Update reservation baggage counts and charge for extra bags.",
        arguments: UPDATE_BAGGAGES_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "update_reservation_flights",
        description: "Replace reservation flights and charge the price difference.",
        arguments: UPDATE_FLIGHTS_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "update_reservation_passengers",
        description: "Replace passenger details when passenger count is unchanged.",
        arguments: UPDATE_PASSENGERS_ARGS,
    },
    ToolSpec {
        requestor: ToolRequestor::Assistant,
        name: "get_flight_status",
        description: "Read the current status of a specific flight instance.",
        arguments: FLIGHT_STATUS_ARGS,
    },
];

pub struct AirlineEnv {
    policy: String,
    db: Value,
}

impl AirlineEnv {
    pub fn load(dataset_root: &Path) -> Result<Self, String> {
        let base = dataset_root.join("tau_bench").join("airline");
        let db = sonic_rs::from_str::<Value>(
            &std::fs::read_to_string(base.join("db.json")).map_err(|err| err.to_string())?,
        )
        .map_err(|err| format!("failed to parse airline db.json: {err}"))?;
        let policy =
            std::fs::read_to_string(base.join("policy.md")).map_err(|err| err.to_string())?;
        Ok(Self { policy, db })
    }

    fn users(&self) -> Result<&Map, String> {
        let root = as_object(&self.db)?;
        let users = get_value(root, "users")?;
        as_object(users)
    }

    fn users_mut(&mut self) -> Result<&mut Map, String> {
        let root = as_object_mut(&mut self.db)?;
        let users = root
            .get_mut(&"users")
            .ok_or_else(|| "missing users in airline db".to_string())?;
        as_object_mut(users)
    }

    fn reservations(&self) -> Result<&Map, String> {
        let root = as_object(&self.db)?;
        let reservations = get_value(root, "reservations")?;
        as_object(reservations)
    }

    fn reservations_mut(&mut self) -> Result<&mut Map, String> {
        let root = as_object_mut(&mut self.db)?;
        let reservations = root
            .get_mut(&"reservations")
            .ok_or_else(|| "missing reservations in airline db".to_string())?;
        as_object_mut(reservations)
    }

    fn flights(&self) -> Result<&Map, String> {
        let root = as_object(&self.db)?;
        let flights = get_value(root, "flights")?;
        as_object(flights)
    }

    fn flights_mut(&mut self) -> Result<&mut Map, String> {
        let root = as_object_mut(&mut self.db)?;
        let flights = root
            .get_mut(&"flights")
            .ok_or_else(|| "missing flights in airline db".to_string())?;
        as_object_mut(flights)
    }

    fn get_user(&self, user_id: &str) -> Result<&Value, String> {
        self.users()?
            .get(&user_id)
            .ok_or_else(|| format!("User {user_id} not found"))
    }

    fn get_user_clone(&self, user_id: &str) -> Result<Value, String> {
        Ok(self.get_user(user_id)?.clone())
    }

    fn set_user(&mut self, user_id: &str, user: Value) -> Result<(), String> {
        self.users_mut()?.insert(&user_id, user);
        Ok(())
    }

    fn get_reservation(&self, reservation_id: &str) -> Result<&Value, String> {
        self.reservations()?
            .get(&reservation_id)
            .ok_or_else(|| format!("Reservation {reservation_id} not found"))
    }

    fn get_reservation_clone(&self, reservation_id: &str) -> Result<Value, String> {
        Ok(self.get_reservation(reservation_id)?.clone())
    }

    fn set_reservation(&mut self, reservation_id: &str, reservation: Value) -> Result<(), String> {
        self.reservations_mut()?
            .insert(&reservation_id, reservation);
        Ok(())
    }

    fn get_flight(&self, flight_number: &str) -> Result<&Value, String> {
        self.flights()?
            .get(&flight_number)
            .ok_or_else(|| format!("Flight {flight_number} not found"))
    }

    fn get_flight_clone(&self, flight_number: &str) -> Result<Value, String> {
        Ok(self.get_flight(flight_number)?.clone())
    }

    fn set_flight(&mut self, flight_number: &str, flight: Value) -> Result<(), String> {
        self.flights_mut()?.insert(&flight_number, flight);
        Ok(())
    }

    fn get_flight_instance<'a>(
        flight: &'a Map,
        flight_number: &str,
        date: &str,
    ) -> Result<&'a Value, String> {
        let dates = as_object(
            flight
                .get(&"dates")
                .ok_or_else(|| format!("missing dates for flight {flight_number}"))?,
        )?;
        dates
            .get(&date)
            .ok_or_else(|| format!("Flight {flight_number} not found on date {date}"))
    }

    fn get_flight_instance_mut<'a>(
        flight: &'a mut Map,
        flight_number: &str,
        date: &str,
    ) -> Result<&'a mut Value, String> {
        let dates = as_object_mut(
            flight
                .get_mut(&"dates")
                .ok_or_else(|| format!("missing dates for flight {flight_number}"))?,
        )?;
        dates
            .get_mut(&date)
            .ok_or_else(|| format!("Flight {flight_number} not found on date {date}"))
    }

    fn get_new_reservation_id(&self) -> Result<&'static str, String> {
        for reservation_id in ["HATHAT", "HATHAU", "HATHAV"] {
            if !self.reservations()?.contains_key(&reservation_id) {
                return Ok(reservation_id);
            }
        }
        Err("Too many reservations".to_string())
    }

    fn get_new_certificate_ids() -> [&'static str; 3] {
        [
            "certificate_3221322",
            "certificate_3221323",
            "certificate_3221324",
        ]
    }

    fn get_datetime() -> &'static str {
        "2024-05-15T15:00:00"
    }

    fn payment_for_update(
        &self,
        user: &mut Value,
        payment_id: &str,
        total_price: i64,
    ) -> Result<Option<Value>, String> {
        let user_object = as_object_mut(user)?;
        let payment_methods = as_object_mut(
            user_object
                .get_mut(&"payment_methods")
                .ok_or_else(|| "missing payment_methods".to_string())?,
        )?;
        let payment_method = payment_methods
            .get_mut(&payment_id)
            .ok_or_else(|| "Payment method not found".to_string())?;
        let payment_object = as_object_mut(payment_method)?;
        let source = get_string_field(payment_object, "source")?;
        if source == "certificate" {
            return Err("Certificate cannot be used to update reservation".to_string());
        }
        if source == "gift_card" {
            let amount = get_f64_field(payment_object, "amount")?;
            if amount < total_price as f64 {
                return Err("Gift card balance is not enough".to_string());
            }
            payment_object.insert(&"amount", json!(amount - total_price as f64));
        }
        if total_price == 0 {
            return Ok(None);
        }
        Ok(Some(json!({
            "payment_id": payment_id,
            "amount": total_price,
        })))
    }

    fn search_direct_internal(
        &self,
        date: &str,
        origin: Option<&str>,
        destination: Option<&str>,
        leave_after: Option<&str>,
    ) -> Result<Vec<Value>, String> {
        let mut results = Vec::new();
        for (_, flight) in self.flights()?.iter() {
            let flight_object = as_object(flight)?;
            let flight_origin = get_string_field(flight_object, "origin")?;
            let flight_destination = get_string_field(flight_object, "destination")?;
            if origin.is_some_and(|value| value != flight_origin) {
                continue;
            }
            if destination.is_some_and(|value| value != flight_destination) {
                continue;
            }
            let date_value = Self::get_flight_instance(
                flight_object,
                get_string_field(flight_object, "flight_number")?,
                date,
            )?;
            let date_object = as_object(date_value)?;
            if get_string_field(date_object, "status")? != "available" {
                continue;
            }
            let departure = get_string_field(flight_object, "scheduled_departure_time_est")?;
            if leave_after.is_some_and(|value| departure < value) {
                continue;
            }
            results.push(json!({
                "flight_number": get_string_field(flight_object, "flight_number")?,
                "origin": flight_origin,
                "destination": flight_destination,
                "status": "available",
                "scheduled_departure_time_est": departure,
                "scheduled_arrival_time_est": get_string_field(flight_object, "scheduled_arrival_time_est")?,
                "available_seats": get_value(date_object, "available_seats")?,
                "prices": get_value(date_object, "prices")?,
            }));
        }
        Ok(results)
    }

    fn execute_airline_tool(&mut self, tool_call: &FunctionCall) -> Result<Value, String> {
        let args = &tool_call.arguments;
        match tool_call.name.as_str() {
            "book_reservation" => self.book_reservation(args),
            "calculate" => Ok(json!(calculate_expression(get_arg_str(
                args,
                "expression",
            )?)?)),
            "cancel_reservation" => self.cancel_reservation(get_arg_str(args, "reservation_id")?),
            "get_reservation_details" => {
                Ok(self.get_reservation_clone(get_arg_str(args, "reservation_id")?)?)
            }
            "get_user_details" => Ok(self.get_user_clone(get_arg_str(args, "user_id")?)?),
            "list_all_airports" => Ok(json!(
                AIRPORTS
                    .iter()
                    .map(|(iata, city)| json!({ "iata": iata, "city": city }))
                    .collect::<Vec<_>>()
            )),
            "search_direct_flight" => Ok(json!(self.search_direct_internal(
                get_arg_str(args, "date")?,
                Some(get_arg_str(args, "origin")?),
                Some(get_arg_str(args, "destination")?),
                None,
            )?)),
            "search_onestop_flight" => self.search_onestop_flight(args),
            "send_certificate" => {
                self.send_certificate(get_arg_str(args, "user_id")?, get_arg_i64(args, "amount")?)
            }
            "transfer_to_human_agents" => Ok(json!("Transfer successful".to_string())),
            "update_reservation_baggages" => self.update_reservation_baggages(args),
            "update_reservation_flights" => self.update_reservation_flights(args),
            "update_reservation_passengers" => self.update_reservation_passengers(args),
            "get_flight_status" => {
                let flight = self.get_flight(get_arg_str(args, "flight_number")?)?;
                let flight_object = as_object(flight)?;
                let date_value = Self::get_flight_instance(
                    flight_object,
                    get_arg_str(args, "flight_number")?,
                    get_arg_str(args, "date")?,
                )?;
                Ok(json!(
                    get_string_field(as_object(date_value)?, "status")?.to_string()
                ))
            }
            other => Err(format!("unsupported airline tool `{other}`")),
        }
    }

    fn book_reservation(&mut self, args: &Map) -> Result<Value, String> {
        let user_id = get_arg_str(args, "user_id")?;
        let origin = get_arg_str(args, "origin")?;
        let destination = get_arg_str(args, "destination")?;
        let flight_type = get_arg_str(args, "flight_type")?;
        let cabin = get_arg_str(args, "cabin")?;
        let flights = get_arg_array(args, "flights")?;
        let passengers = get_arg_array(args, "passengers")?;
        let payment_methods = get_arg_array(args, "payment_methods")?;
        let total_baggages = get_arg_i64(args, "total_baggages")?;
        let nonfree_baggages = get_arg_i64(args, "nonfree_baggages")?;
        let insurance = get_arg_str(args, "insurance")?;

        let mut user = self.get_user_clone(user_id)?;
        let reservation_id = self.get_new_reservation_id()?.to_string();
        let mut reservation_flights = Vec::new();
        let passenger_count = passengers.len() as i64;
        let mut total_price = 0_i64;
        let mut seat_updates: Vec<(String, String)> = Vec::new();

        for flight_info in flights {
            let info = as_object(flight_info)?;
            let flight_number = get_string_field(info, "flight_number")?;
            let date = get_string_field(info, "date")?;
            let flight = self.get_flight_clone(flight_number)?;
            let flight_object = as_object(&flight)?;
            let date_value = Self::get_flight_instance(flight_object, flight_number, date)?;
            let date_object = as_object(date_value)?;
            if get_string_field(date_object, "status")? != "available" {
                return Err(format!(
                    "Flight {flight_number} not available on date {date}"
                ));
            }
            let available = as_object(
                date_object
                    .get(&"available_seats")
                    .ok_or_else(|| "missing available_seats".to_string())?,
            )?;
            let available_count = available
                .get(&cabin)
                .and_then(Value::as_i64)
                .or_else(|| {
                    available
                        .get(&cabin)
                        .and_then(Value::as_u64)
                        .map(|v| v as i64)
                })
                .ok_or_else(|| format!("missing seat count for cabin `{cabin}`"))?;
            if available_count < passenger_count {
                return Err(format!("Not enough seats on flight {flight_number}"));
            }
            let prices = as_object(
                date_object
                    .get(&"prices")
                    .ok_or_else(|| "missing prices".to_string())?,
            )?;
            let price = prices
                .get(&cabin)
                .and_then(Value::as_i64)
                .or_else(|| prices.get(&cabin).and_then(Value::as_u64).map(|v| v as i64))
                .ok_or_else(|| format!("missing price for cabin `{cabin}`"))?;
            total_price += price * passenger_count;
            seat_updates.push((flight_number.to_string(), date.to_string()));
            reservation_flights.push(json!({
                "origin": get_string_field(flight_object, "origin")?,
                "destination": get_string_field(flight_object, "destination")?,
                "flight_number": flight_number,
                "date": date,
                "price": price,
            }));
        }

        if insurance == "yes" {
            total_price += 30 * passenger_count;
        }
        total_price += 50 * nonfree_baggages;

        let user_object = as_object_mut(&mut user)?;
        let payment_method_map = as_object_mut(
            user_object
                .get_mut(&"payment_methods")
                .ok_or_else(|| "missing payment_methods".to_string())?,
        )?;
        let mut total_payment = 0_i64;
        for payment in payment_methods {
            let payment_object = as_object(payment)?;
            let payment_id = get_string_field(payment_object, "payment_id")?;
            let amount = payment_object
                .get(&"amount")
                .and_then(Value::as_i64)
                .or_else(|| {
                    payment_object
                        .get(&"amount")
                        .and_then(Value::as_u64)
                        .map(|v| v as i64)
                })
                .ok_or_else(|| "missing payment amount".to_string())?;
            let user_payment_method = payment_method_map
                .get(&payment_id)
                .ok_or_else(|| format!("Payment method {payment_id} not found"))?;
            let payment_method_object = as_object(user_payment_method)?;
            let source = get_string_field(payment_method_object, "source")?;
            if matches!(source, "gift_card" | "certificate") {
                let balance = get_f64_field(payment_method_object, "amount")?;
                if balance < amount as f64 {
                    return Err(format!("Not enough balance in payment method {payment_id}"));
                }
            }
            total_payment += amount;
        }
        if total_payment != total_price {
            return Err(format!(
                "Payment amount does not add up, total price is {total_price}, but paid {total_payment}"
            ));
        }

        for payment in payment_methods {
            let payment_object = as_object(payment)?;
            let payment_id = get_string_field(payment_object, "payment_id")?;
            let amount = payment_object
                .get(&"amount")
                .and_then(Value::as_i64)
                .or_else(|| {
                    payment_object
                        .get(&"amount")
                        .and_then(Value::as_u64)
                        .map(|v| v as i64)
                })
                .ok_or_else(|| "missing payment amount".to_string())?;
            let source = {
                let method = payment_method_map
                    .get(&payment_id)
                    .ok_or_else(|| format!("Payment method {payment_id} not found"))?;
                get_string_field(as_object(method)?, "source")?.to_string()
            };
            match source.as_str() {
                "gift_card" => {
                    let method = payment_method_map
                        .get_mut(&payment_id)
                        .ok_or_else(|| format!("Payment method {payment_id} not found"))?;
                    let method_object = as_object_mut(method)?;
                    let balance = get_f64_field(method_object, "amount")?;
                    method_object.insert(&"amount", json!(balance - amount as f64));
                }
                "certificate" => {
                    payment_method_map.remove(&payment_id);
                }
                _ => {}
            }
        }

        if let Some(user_reservations) = user_object.get_mut(&"reservations") {
            as_array_mut(user_reservations)?.push(json!(reservation_id.clone()));
        }
        self.set_user(user_id, user)?;

        for (flight_number, date) in seat_updates {
            let mut flight = self.get_flight_clone(&flight_number)?;
            let flight_object = as_object_mut(&mut flight)?;
            let date_value = Self::get_flight_instance_mut(flight_object, &flight_number, &date)?;
            let date_object = as_object_mut(date_value)?;
            let available_seats = as_object_mut(
                date_object
                    .get_mut(&"available_seats")
                    .ok_or_else(|| "missing available_seats".to_string())?,
            )?;
            let available = available_seats
                .get(&cabin)
                .and_then(Value::as_i64)
                .or_else(|| {
                    available_seats
                        .get(&cabin)
                        .and_then(Value::as_u64)
                        .map(|v| v as i64)
                })
                .ok_or_else(|| format!("missing seat count for cabin `{cabin}`"))?;
            available_seats.insert(&cabin, json!(available - passenger_count));
            self.set_flight(&flight_number, flight)?;
        }

        let reservation = json!({
            "reservation_id": reservation_id,
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": flight_type,
            "cabin": cabin,
            "flights": reservation_flights,
            "passengers": passengers,
            "payment_history": payment_methods,
            "created_at": Self::get_datetime(),
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
        });
        self.set_reservation(
            reservation
                .get(&"reservation_id")
                .and_then(Value::as_str)
                .unwrap(),
            reservation.clone(),
        )?;
        Ok(reservation)
    }

    fn cancel_reservation(&mut self, reservation_id: &str) -> Result<Value, String> {
        let mut reservation = self.get_reservation_clone(reservation_id)?;
        let reservation_object = as_object_mut(&mut reservation)?;
        let refunds = {
            let payments = as_array(
                reservation_object
                    .get(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?;
            payments
                .iter()
                .map(|payment| {
                    let payment_object = as_object(payment)?;
                    Ok(json!({
                        "payment_id": get_string_field(payment_object, "payment_id")?,
                        "amount": -payment_object
                            .get(&"amount")
                            .and_then(Value::as_i64)
                            .or_else(|| payment_object.get(&"amount").and_then(Value::as_u64).map(|v| v as i64))
                            .ok_or_else(|| "missing payment amount".to_string())?,
                    }))
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        as_array_mut(
            reservation_object
                .get_mut(&"payment_history")
                .ok_or_else(|| "missing payment_history".to_string())?,
        )?
        .extend(refunds.iter());
        reservation_object.insert(&"status", json!("cancelled".to_string()));
        self.set_reservation(reservation_id, reservation.clone())?;
        Ok(reservation)
    }

    fn send_certificate(&mut self, user_id: &str, amount: i64) -> Result<Value, String> {
        let mut user = self.get_user_clone(user_id)?;
        let user_object = as_object_mut(&mut user)?;
        let payment_methods = as_object_mut(
            user_object
                .get_mut(&"payment_methods")
                .ok_or_else(|| "missing payment_methods".to_string())?,
        )?;
        for payment_id in Self::get_new_certificate_ids() {
            if !payment_methods.contains_key(&payment_id) {
                payment_methods.insert(
                    &payment_id,
                    json!({
                        "id": payment_id,
                        "amount": amount,
                        "source": "certificate",
                    }),
                );
                self.set_user(user_id, user)?;
                return Ok(json!(format!(
                    "Certificate {payment_id} added to user {user_id} with amount {amount}."
                )));
            }
        }
        Err("Too many certificates".to_string())
    }

    fn update_reservation_baggages(&mut self, args: &Map) -> Result<Value, String> {
        let reservation_id = get_arg_str(args, "reservation_id")?;
        let total_baggages = get_arg_i64(args, "total_baggages")?;
        let nonfree_baggages = get_arg_i64(args, "nonfree_baggages")?;
        let payment_id = get_arg_str(args, "payment_id")?;

        let mut reservation = self.get_reservation_clone(reservation_id)?;
        let reservation_object = as_object(&reservation)?;
        let old_nonfree = reservation_object
            .get(&"nonfree_baggages")
            .and_then(Value::as_i64)
            .or_else(|| {
                reservation_object
                    .get(&"nonfree_baggages")
                    .and_then(Value::as_u64)
                    .map(|v| v as i64)
            })
            .ok_or_else(|| "missing nonfree_baggages".to_string())?;
        let total_price = 50 * std::cmp::max(0, nonfree_baggages - old_nonfree);
        let user_id = get_string_field(reservation_object, "user_id")?.to_string();

        let mut user = self.get_user_clone(&user_id)?;
        if let Some(payment) = self.payment_for_update(&mut user, payment_id, total_price)? {
            let reservation_object = as_object_mut(&mut reservation)?;
            as_array_mut(
                reservation_object
                    .get_mut(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?
            .push(payment);
        }
        self.set_user(&user_id, user)?;
        let reservation_object = as_object_mut(&mut reservation)?;
        reservation_object.insert(&"total_baggages", json!(total_baggages));
        reservation_object.insert(&"nonfree_baggages", json!(nonfree_baggages));
        self.set_reservation(reservation_id, reservation.clone())?;
        Ok(reservation)
    }

    fn update_reservation_flights(&mut self, args: &Map) -> Result<Value, String> {
        let reservation_id = get_arg_str(args, "reservation_id")?;
        let cabin = get_arg_str(args, "cabin")?;
        let flights = get_arg_array(args, "flights")?;
        let payment_id = get_arg_str(args, "payment_id")?;

        let mut reservation = self.get_reservation_clone(reservation_id)?;
        let reservation_object_ro = as_object(&reservation)?;
        let old_cabin = get_string_field(reservation_object_ro, "cabin")?.to_string();
        let existing_flights = as_array(
            reservation_object_ro
                .get(&"flights")
                .ok_or_else(|| "missing flights".to_string())?,
        )?
        .clone();
        let passenger_count = as_array(
            reservation_object_ro
                .get(&"passengers")
                .ok_or_else(|| "missing passengers".to_string())?,
        )?
        .len() as i64;
        let user_id = get_string_field(reservation_object_ro, "user_id")?.to_string();

        let mut total_price = 0_i64;
        let mut reservation_flights = Vec::new();
        for flight_info in flights {
            let info = as_object(flight_info)?;
            let flight_number = get_string_field(info, "flight_number")?;
            let date = get_string_field(info, "date")?;

            if let Some(existing) = existing_flights.iter().find(|value| {
                let value_object = as_object(value).ok();
                value_object.is_some_and(|obj| {
                    obj.get(&"flight_number").and_then(Value::as_str) == Some(flight_number)
                        && obj.get(&"date").and_then(Value::as_str) == Some(date)
                        && old_cabin == cabin
                })
            }) {
                let price = as_object(existing)?
                    .get(&"price")
                    .and_then(Value::as_i64)
                    .or_else(|| {
                        as_object(existing)
                            .ok()?
                            .get(&"price")
                            .and_then(Value::as_u64)
                            .map(|v| v as i64)
                    })
                    .ok_or_else(|| "missing reservation flight price".to_string())?;
                total_price += price * passenger_count;
                reservation_flights.push(existing.clone());
                continue;
            }

            let flight = self.get_flight_clone(flight_number)?;
            let flight_object = as_object(&flight)?;
            let date_value = Self::get_flight_instance(flight_object, flight_number, date)?;
            let date_object = as_object(date_value)?;
            if get_string_field(date_object, "status")? != "available" {
                return Err(format!(
                    "Flight {flight_number} not available on date {date}"
                ));
            }
            let available = as_object(
                date_object
                    .get(&"available_seats")
                    .ok_or_else(|| "missing available_seats".to_string())?,
            )?;
            let available_count = available
                .get(&cabin)
                .and_then(Value::as_i64)
                .or_else(|| {
                    available
                        .get(&cabin)
                        .and_then(Value::as_u64)
                        .map(|v| v as i64)
                })
                .ok_or_else(|| format!("missing seat count for cabin `{cabin}`"))?;
            if available_count < passenger_count {
                return Err(format!("Not enough seats on flight {flight_number}"));
            }
            let prices = as_object(
                date_object
                    .get(&"prices")
                    .ok_or_else(|| "missing prices".to_string())?,
            )?;
            let price = prices
                .get(&cabin)
                .and_then(Value::as_i64)
                .or_else(|| prices.get(&cabin).and_then(Value::as_u64).map(|v| v as i64))
                .ok_or_else(|| format!("missing price for cabin `{cabin}`"))?;
            total_price += price * passenger_count;
            reservation_flights.push(json!({
                "flight_number": flight_number,
                "date": date,
                "price": price,
                "origin": get_string_field(flight_object, "origin")?,
                "destination": get_string_field(flight_object, "destination")?,
            }));
        }

        let already_paid = existing_flights
            .iter()
            .map(|flight| {
                as_object(flight)?
                    .get(&"price")
                    .and_then(Value::as_i64)
                    .or_else(|| {
                        as_object(flight)
                            .ok()?
                            .get(&"price")
                            .and_then(Value::as_u64)
                            .map(|v| v as i64)
                    })
                    .ok_or_else(|| "missing reservation flight price".to_string())
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .sum::<i64>()
            * passenger_count;
        total_price -= already_paid;

        let mut user = self.get_user_clone(&user_id)?;
        if let Some(payment) = self.payment_for_update(&mut user, payment_id, total_price)? {
            let reservation_object = as_object_mut(&mut reservation)?;
            as_array_mut(
                reservation_object
                    .get_mut(&"payment_history")
                    .ok_or_else(|| "missing payment_history".to_string())?,
            )?
            .push(payment);
        }
        self.set_user(&user_id, user)?;
        let reservation_object = as_object_mut(&mut reservation)?;
        reservation_object.insert(&"flights", json!(reservation_flights));
        reservation_object.insert(&"cabin", json!(cabin.to_string()));
        self.set_reservation(reservation_id, reservation.clone())?;
        Ok(reservation)
    }

    fn update_reservation_passengers(&mut self, args: &Map) -> Result<Value, String> {
        let reservation_id = get_arg_str(args, "reservation_id")?;
        let passengers = get_arg_array(args, "passengers")?;
        let mut reservation = self.get_reservation_clone(reservation_id)?;
        let reservation_object = as_object_mut(&mut reservation)?;
        let existing_len = as_array(
            reservation_object
                .get(&"passengers")
                .ok_or_else(|| "missing passengers".to_string())?,
        )?
        .len();
        if passengers.len() != existing_len {
            return Err("Number of passengers does not match".to_string());
        }
        reservation_object.insert(&"passengers", json!(passengers.to_vec()));
        self.set_reservation(reservation_id, reservation.clone())?;
        Ok(reservation)
    }

    fn search_onestop_flight(&self, args: &Map) -> Result<Value, String> {
        let origin = get_arg_str(args, "origin")?;
        let destination = get_arg_str(args, "destination")?;
        let date = get_arg_str(args, "date")?;
        let mut results = Vec::new();
        for mut result1 in self.search_direct_internal(date, Some(origin), None, None)? {
            let departure_date = if result1
                .get(&"scheduled_arrival_time_est")
                .and_then(Value::as_str)
                .is_some_and(|value| value.contains("+1"))
            {
                increment_may_2024_date(date)?
            } else {
                date.to_string()
            };
            if let Some(result1_obj) = result1.as_object_mut() {
                result1_obj.insert(&"date", json!(date.to_string()));
            }
            let leave_after = result1
                .get(&"scheduled_arrival_time_est")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing scheduled_arrival_time_est".to_string())?
                .to_string();
            let leg_origin = result1
                .get(&"destination")
                .and_then(Value::as_str)
                .ok_or_else(|| "missing destination".to_string())?
                .to_string();
            for mut result2 in self.search_direct_internal(
                &departure_date,
                Some(&leg_origin),
                Some(destination),
                Some(&leave_after),
            )? {
                if let Some(result2_obj) = result2.as_object_mut() {
                    result2_obj.insert(&"date", json!(departure_date.clone()));
                }
                results.push(json!(vec![result1.clone(), result2]));
            }
        }
        Ok(json!(results))
    }
}

impl TauDomainEnv for AirlineEnv {
    fn policy(&self) -> &str {
        &self.policy
    }

    fn assistant_tools(&self) -> &'static [ToolSpec] {
        AIRLINE_ASSISTANT_TOOLS
    }

    fn user_tools(&self) -> &'static [ToolSpec] {
        &[]
    }

    fn update_agent_data(&mut self, data: &Value) -> Result<(), String> {
        update_json(&mut self.db, data)
    }

    fn update_user_data(&mut self, _data: &Value) -> Result<(), String> {
        Err("airline has no separate user db".to_string())
    }

    fn run_env_function(&mut self, action: &EnvFunctionCall) -> Result<Value, String> {
        if action.env_type != ToolRequestor::Assistant {
            return Err("airline only supports assistant env functions".to_string());
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
            return Err("airline does not support user tool calls".to_string());
        }
        self.execute_airline_tool(tool_call)
    }

    fn run_env_assertion(&self, _assertion: &EnvAssertion) -> Result<bool, String> {
        Err("airline has no env assertions".to_string())
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

fn get_arg_i64(args: &Map, key: &str) -> Result<i64, String> {
    args.get(&key)
        .and_then(Value::as_i64)
        .or_else(|| args.get(&key).and_then(Value::as_u64).map(|v| v as i64))
        .ok_or_else(|| format!("missing integer argument `{key}`"))
}

fn get_arg_array<'a>(args: &'a Map, key: &str) -> Result<&'a [Value], String> {
    Ok(args
        .get(&key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array argument `{key}`"))?)
}

fn increment_may_2024_date(date: &str) -> Result<String, String> {
    let day = date
        .strip_prefix("2024-05-")
        .ok_or_else(|| format!("unsupported airline date `{date}`"))?
        .parse::<u32>()
        .map_err(|err| format!("invalid airline date `{date}`: {err}"))?;
    Ok(format!("2024-05-{:02}", day + 1))
}
