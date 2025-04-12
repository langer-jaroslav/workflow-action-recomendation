import os
import random
import pandas as pd
import numpy as np
from enum import Enum
rows_to_generate = 10000
file_path = "./data/requests.csv"

# Enums
class RequestPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RequestStatus(Enum):
    APPROVED = 'Approved'
    REJECTED = 'Rejected'

class OrderType(Enum):
    MATERIAL_FOR_PRODUCTION = "material_for_production"
    OFFICE_EQUIPMENT = "office_equipment"
    OTHER = "other"

class EmployeePosition(Enum):
    SECRETARY = "secretary"
    TECHNICIAN = "technician"
    ENGINEER = "engineer"
    MANAGER = "manager"
    PURCHASING_SPECIALIST = "purchasing_specialist"

position_order_type_weights = {
    EmployeePosition.SECRETARY: [0.0, 0.8, 0.2],
    EmployeePosition.TECHNICIAN: [0.6, 0.2, 0.2],
    EmployeePosition.ENGINEER: [0.7, 0.2, 0.1],
    EmployeePosition.MANAGER: [0.2, 0.5, 0.3],
    EmployeePosition.PURCHASING_SPECIALIST: [0.5, 0.3, 0.2]
}

def categorize_risk(risk_score):
    if risk_score <= 0:
        return "low"
    elif 1 <= risk_score <= 4:
        return "medium"
    else:
        return "high"

def determine_status(risk_score):
    if risk_score >= 5:
        return RequestStatus.REJECTED
    elif risk_score <= -1:
        return RequestStatus.APPROVED
    else:
        return random.choices([RequestStatus.APPROVED, RequestStatus.REJECTED], weights=[0.6, 0.4])[0]

num_employees = 200
employee_ids = [f"EMP{str(i).zfill(4)}" for i in range(1, num_employees + 1)]
employee_positions = {
    emp_id: random.choice(list(EmployeePosition))
    for emp_id in employee_ids
}

def generate_random_request(request_id):
    employee_id = random.choice(employee_ids)
    position = employee_positions[employee_id]

    requested_items = random.randint(5, 40)
    priority = random.choices(list(RequestPriority), weights=[0.3, 0.5, 0.2])[0]

    is_urgent = random.choices(
        [True, False],
        weights=[0.8, 0.2] if priority == RequestPriority.HIGH else
                [0.2, 0.8] if priority == RequestPriority.LOW else
                [0.4, 0.6])[0]

    price_per_item = round(random.uniform(50, 400), 2)
    total_value = round(price_per_item * requested_items, 2)
    request_age = random.randint(1, 60)

    order_type = random.choices(
        list(OrderType),
        weights=position_order_type_weights[position]
    )[0]

    risk_score = 0

    if total_value > 5000:
        risk_score += 2
    elif total_value < 2000:
        risk_score -= 1

    if priority == RequestPriority.LOW:
        risk_score += 3
    elif priority == RequestPriority.HIGH:
        risk_score -= 2

    if requested_items > 30:
        risk_score += 2
    elif requested_items < 10:
        risk_score -= 1

    if request_age > 30:
        risk_score += 2
    elif request_age < 10:
        risk_score -= 1

    if price_per_item > 300:
        risk_score += 2
    elif price_per_item < 100:
        risk_score -= 1

    if is_urgent:
        risk_score -= 1

    if order_type == OrderType.OTHER:
        risk_score += 2
    elif order_type == OrderType.OFFICE_EQUIPMENT:
        risk_score += 1
        if total_value > 6000:
            risk_score += 1
    elif order_type == OrderType.MATERIAL_FOR_PRODUCTION:
        risk_score -= 2

    risk_score += random.choice([-1, 0, 1])

    status = determine_status(risk_score)
    risk_category = categorize_risk(risk_score)

    return {
        "request_id": request_id,
        "employee_id": employee_id,
        "employee_position": position.value,
        "requested_items": requested_items,
        "status": status.value,
        "priority": priority.value,
        "is_urgent": is_urgent,
        "total_value": total_value,
        "price_per_item": price_per_item,
        "order_type": order_type.value,
        "request_age": request_age,
        "risk_score": risk_score,
        "risk_score_category": risk_category
    }

def generate_requests_list(num_requests):
    requests = []
    approved_count = 0
    rejected_count = 0
    target_count = num_requests // 2

    for i in range(1, num_requests + 1):
        request = generate_random_request(i)

        if request["status"] == "Approved":
            if approved_count < target_count:
                approved_count += 1
            else:
                request["status"] = "Rejected"
                rejected_count += 1
        else:
            if rejected_count < target_count:
                rejected_count += 1
            else:
                request["status"] = "Approved"
                approved_count += 1

        requests.append(request)

    return requests

# Export CSV
def export_to_csv(requests, filename):
    df = pd.DataFrame(requests)
    features_to_square = ["requested_items", "total_value", "price_per_item", "request_age", "risk_score"]
    for feature in features_to_square:
        df[f"{feature}_squared"] = np.power(df[feature], 2)
    df.to_csv(filename, index=False)

requests_list = generate_requests_list(rows_to_generate)
if os.path.exists(file_path):
    os.remove(file_path)
export_to_csv(requests_list, file_path)

print(f"Clean dataset generated and saved to {file_path}")
