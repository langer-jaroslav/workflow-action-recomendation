import os
import random
import pandas as pd
import numpy as np
from enum import Enum

# Number of rows to generate
rows_to_generate = 10000
file_path = "./data/requests.csv"

# Enums for categorical variables
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

# Function to determine request status based on a more deterministic risk model
def determine_status(risk_score):
    """
    Determines whether a request is approved or rejected based on its calculated risk score.
    """
    if risk_score >= 5:
        return RequestStatus.REJECTED
    elif risk_score <= -1:
        return RequestStatus.APPROVED
    else:
        return random.choices([RequestStatus.APPROVED, RequestStatus.REJECTED], weights=[0.6, 0.4])[0]

# Generate a single request
def generate_random_request(request_id):
    """
    Generates a single request with random attributes and calculates the risk score.
    """
    requested_items = random.randint(5, 40)
    priority = random.choices(list(RequestPriority), weights=[0.3, 0.5, 0.2], k=1)[0]
    is_urgent = random.choices([True, False], weights=[0.4, 0.6], k=1)[0]
    is_from_wholesaler = random.choices([True, False], weights=[0.6, 0.4], k=1)[0]
    total_value = round(random.uniform(1500, 7000), 2)
    price_per_item = total_value / requested_items
    request_age = random.randint(1, 60)  # New Feature: Age of the request
    order_type = random.choices(list(OrderType), weights=[0.5, 0.3, 0.2], k=1)[0]

    # **Compute Risk Score**
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

    if is_from_wholesaler:
        risk_score -= 1
    else:
        risk_score += 1

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

    # **Impact of Order Type**
    if order_type == OrderType.OTHER:
        risk_score += 2  # Generic orders are riskier
    elif order_type == OrderType.OFFICE_EQUIPMENT:
        risk_score += 1  # Office equipment is moderately risky
    elif order_type == OrderType.MATERIAL_FOR_PRODUCTION:
        risk_score -= 2  # Production material is safer

    # Determine status based on risk score
    status = determine_status(risk_score)
    
    return {
        "request_id": request_id,
        "requested_items": requested_items,
        "status": status.value,
        "priority": priority.value,
        "is_urgent": is_urgent,
        "is_from_wholesaler": is_from_wholesaler,
        "total_value": total_value,
        "price_per_item": round(price_per_item, 2),
        "order_type": order_type.value,
        "request_age": request_age,  # New Feature
        "risk_score": risk_score  # Store risk score to analyze later
    }

# Generate a balanced dataset
def generate_requests_list(num_requests):
    """
    Generates a dataset of requests, ensuring a balanced distribution of approvals and rejections.
    """
    requests = []
    approved_count = 0
    rejected_count = 0
    target_count = num_requests // 2  # Balance 50/50

    for i in range(1, num_requests + 1):
        request = generate_random_request(i)

        if request["status"] == "Approved":
            if approved_count < target_count:
                approved_count += 1
                requests.append(request)
            else:
                request["status"] = "Rejected"
                rejected_count += 1
                requests.append(request)
        elif request["status"] == "Rejected":
            if rejected_count < target_count:
                rejected_count += 1
                requests.append(request)
            else:
                request["status"] = "Approved"
                approved_count += 1
                requests.append(request)

    return requests

# Export to CSV
def export_to_csv(requests, filename):
    df = pd.DataFrame(requests)

    # **Square important numerical features**
    features_to_square = ["requested_items", "total_value", "price_per_item", "request_age", "risk_score"]
    for feature in features_to_square:
        df[f"{feature}_squared"] = np.power(df[feature], 2)

    df.to_csv(filename, index=False)

# Run data generation
requests_list = generate_requests_list(rows_to_generate)
if os.path.exists(file_path):
    os.remove(file_path)
export_to_csv(requests_list, file_path)

print(f"✅ New dataset generated and saved to {file_path} with squared features included!")
