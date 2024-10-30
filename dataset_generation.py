import os
import random
from enum import Enum
import pandas as pd

rows_to_generate = 10000
file_path = "./data/requests.csv"

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

class Request:
    def __init__(self, request_id, requested_items, status, priority, is_urgent, 
                 is_from_wholesaler, total_value, price_per_item, order_type):
        self.request_id = request_id
        self.requested_items = requested_items
        self.status = status
        self.priority = priority
        self.is_urgent = is_urgent
        self.is_from_wholesaler = is_from_wholesaler  
        self.total_value = total_value  
        self.price_per_item = price_per_item
        self.order_type = order_type 

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "requested_items": self.requested_items,
            "status": self.status.value,
            "priority": self.priority.value,
            "is_urgent": self.is_urgent,
            "is_from_wholesaler": self.is_from_wholesaler,
            "total_value": self.total_value,
            "price_per_item": self.price_per_item,
            "order_type": self.order_type.value
        }

    def __repr__(self):
        return (f"Request(request_id={self.request_id}, requested_items={self.requested_items}, "
                f"status={self.status}, priority={self.priority}, is_urgent={self.is_urgent}, "
                f"is_from_wholesaler={self.is_from_wholesaler}, total_value={self.total_value}, "
                f"price_per_item={self.price_per_item}, order_type={self.order_type})")

def determine_status(requested_items, is_urgent, is_from_wholesaler, total_value, priority, 
                     price_per_item, order_type: OrderType):
    # Adjusted initial probabilities for a more even balance
    base_reject_prob = 0.5
    base_approve_prob = 0.5

    # Order type influence
    if order_type == OrderType.MATERIAL_FOR_PRODUCTION:
        base_approve_prob += 0.35
        base_reject_prob -= 0.15
    elif order_type == OrderType.OFFICE_EQUIPMENT:
        base_reject_prob += 0.25
        base_approve_prob += 0.05
    elif order_type == OrderType.OTHER:
        base_reject_prob += 0.15

    # Requested items
    if requested_items > 30:
        base_reject_prob += 0.15
    elif requested_items < 10:
        base_approve_prob += 0.15

    # Total value influence
    if total_value > 5000:
        base_reject_prob += 0.25
    elif total_value < 2000:
        base_approve_prob += 0.25

    # Urgency influence
    if is_urgent:
        base_approve_prob += 0.25
    else:
        base_reject_prob += 0.15

    # Wholesale supplier influence
    if is_from_wholesaler:
        base_approve_prob += 0.25
    else:
        base_reject_prob += 0.25

    # Price per item influence
    if price_per_item > 300:
        base_reject_prob += 0.25
    elif price_per_item < 100:
        base_approve_prob += 0.15

    # Priority influence
    if priority == RequestPriority.HIGH:
        base_approve_prob += 0.35
    elif priority == RequestPriority.LOW:
        base_reject_prob += 0.3

    # Normalizing probabilities for a 50/50 target
    total_prob = base_reject_prob + base_approve_prob
    base_reject_prob /= total_prob
    base_approve_prob /= total_prob

    # Random decision with refined binary probabilities
    random_value = random.random()
    if random_value < base_reject_prob:
        return RequestStatus.REJECTED
    else:
        return RequestStatus.APPROVED



def generate_random_request(request_id):
    requested_items = random.randint(5, 40)
    priority = random.choices(list(RequestPriority), weights=[0.1, 0.6, 0.3], k=1)[0]
    is_urgent = random.choices([True, False], weights=[0.5, 0.5], k=1)[0]
    is_from_wholesaler = random.choices([True, False], weights=[0.7, 0.3], k=1)[0]
    total_value = round(random.uniform(1500, 7000), 2)
    price_per_item = total_value / requested_items
    
    order_type = random.choices(list(OrderType), weights=[0.5, 0.3, 0.2], k=1)[0]

    status = determine_status(requested_items, is_urgent, is_from_wholesaler, total_value, priority, 
                              price_per_item, order_type)
    
    return Request(
        request_id=request_id,
        requested_items=requested_items,
        status=status,
        priority=priority,
        is_urgent=is_urgent,
        is_from_wholesaler=is_from_wholesaler,
        total_value=total_value,
        price_per_item=round(price_per_item, 2),
        order_type=order_type
    )

def generate_requests_list(num_requests):
    requests = []
    approved_count = 0
    rejected_count = 0
    target_count = num_requests // 2  # Target count for each category

    for i in range(1, num_requests + 1):
        request = generate_random_request(i)

        # Track counts to enforce 50/50 split
        if request.status == RequestStatus.APPROVED:
            if approved_count < target_count:
                approved_count += 1
                requests.append(request)
            else:
                # Force "Rejected" if "Approved" limit reached
                request.status = RequestStatus.REJECTED
                rejected_count += 1
                requests.append(request)
        elif request.status == RequestStatus.REJECTED:
            if rejected_count < target_count:
                rejected_count += 1
                requests.append(request)
            else:
                # Force "Approved" if "Rejected" limit reached
                request.status = RequestStatus.APPROVED
                approved_count += 1
                requests.append(request)

    return requests

def export_to_csv(requests, filename):
    fieldnames = ["request_id", "requested_items", "status", "priority", "is_urgent", 
                  "is_from_wholesaler", "total_value", "price_per_item", "order_type"]
    df = pd.DataFrame([r.to_dict() for r in requests])
    df.to_csv(filename, index=False)


requests_list = generate_requests_list(rows_to_generate)
os.remove(file_path)
export_to_csv(requests_list, file_path)
