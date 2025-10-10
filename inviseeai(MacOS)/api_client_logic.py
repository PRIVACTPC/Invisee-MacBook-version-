from api_clients import KNMIClient, DummyAPIClient  # Import your classes from the main logic

# Registry of available clients
_clients_registry = {
    "KNMI": KNMIClient(),
    "Dummy": DummyAPIClient(),
}

def get_available_clients():
    return list(_clients_registry.keys())

def get_client(name):
    return _clients_registry.get(name)