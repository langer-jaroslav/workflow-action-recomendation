import random
from enum import Enum
from datetime import datetime, timedelta
import csv
import json

class RequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class RequestPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DeliveryLocation(Enum):
    CZECH_REPUBLIC = "Czech Republic"
    SLOVAKIA = "Slovakia"
    GERMANY = "Germany"
    AUSTRIA = "Austria"
    POLAND = "Poland"
    HUNGARY = "Hungary"
    FRANCE = "France"
    ITALY = "Italy"

class Request:
    def __init__(self, request_id, requested_items, request_date, status, priority, is_urgent, expected_delivery, is_from_wholesaler, delivery_location, total_value):
        self.request_id = request_id
        self.requested_items = requested_items
        self.request_date = request_date
        self.status = status
        self.priority = priority
        self.is_urgent = is_urgent
        self.expected_delivery = expected_delivery
        self.is_from_wholesaler = is_from_wholesaler  
        self.delivery_location = delivery_location  
        self.total_value = total_value  

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "requested_items": self.requested_items,
            "request_date": self.request_date,
            "status": self.status.value,
            "priority": self.priority.value,
            "is_urgent": self.is_urgent,
            "expected_delivery": self.expected_delivery,
            "is_from_wholesaler": self.is_from_wholesaler,
            "delivery_location": self.delivery_location.value,
            "total_value": self.total_value
        }

    def __repr__(self):
        wholesaler_info = "Yes" if self.is_from_wholesaler else "No"
        return (f"Request(request_id={self.request_id}, requested_items={self.requested_items}, request_date={self.request_date}, "
                f"status={self.status}, priority={self.priority}, is_urgent={self.is_urgent}, "
                f"expected_delivery={self.expected_delivery}, is_from_wholesaler={wholesaler_info}, "
                f"delivery_location={self.delivery_location.value}, total_value={self.total_value})")

# Funkce pro nastavení statusu na základě korelace s ostatními atributy
def determine_status(priority, is_urgent, is_from_wholesaler, total_value):
    # Základní pravděpodobnosti založené na atributech
    if is_from_wholesaler or total_value > 5000:
        # Vysoká šance na schválení pro velké objednávky nebo od velkoobchodníka
        if random.random() < 0.8:
            return RequestStatus.APPROVED
    if priority == RequestPriority.HIGH or is_urgent:
        # Pokud je vysoká priorita nebo urgentní, vyšší šance na pending/approved
        if random.random() < 0.7:
            return RequestStatus.PENDING
        else:
            return RequestStatus.APPROVED
    if priority == RequestPriority.LOW and total_value < 1000:
        # Nízká priorita a malá hodnota, vyšší šance na odmítnutí
        if random.random() < 0.7:
            return RequestStatus.REJECTED
    
    # Anomálie nebo náhodné chování
    return random.choice([RequestStatus.PENDING, RequestStatus.APPROVED, RequestStatus.REJECTED])

def generate_random_request(request_id):
    requested_items = random.randint(1, 50)
    request_date = datetime.now() - timedelta(days=random.randint(1, 30))  
    priority = random.choice(list(RequestPriority))  
    is_urgent = random.choice([True, False]) 
    expected_delivery = request_date + timedelta(days=random.randint(1, 15))  
    is_from_wholesaler = random.choice([True, False])  
    delivery_location = random.choice(list(DeliveryLocation)) 
    total_value = round(random.uniform(500, 10000), 2)  
    
    status = determine_status(priority, is_urgent, is_from_wholesaler, total_value)
    
    return Request(request_id, requested_items, request_date.strftime("%Y-%m-%d"), status, priority, is_urgent, expected_delivery.strftime("%Y-%m-%d"), is_from_wholesaler, delivery_location, total_value)

def generate_requests_list(num_requests):
    requests = []
    for i in range(1, num_requests + 1):
        requests.append(generate_random_request(i))
    return requests

def export_to_csv(requests, filename):
    fieldnames = ["request_id", "requested_items", "request_date", "status", "priority", "is_urgent", "expected_delivery", "is_from_wholesaler", "delivery_location", "total_value"]
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for request in requests:
            writer.writerow(request.to_dict())

def export_to_json(requests, filename):
    with open(filename, 'w') as file:
        json.dump([request.to_dict() for request in requests], file, indent=4)



requests_list = generate_requests_list(1000)
export_to_csv(requests_list, './data/requests.csv')
export_to_json(requests_list, './data/requests.json')
