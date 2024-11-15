from .core import ProvitasCore
from .database import init_db

def main():
    """
    The main entry point of the application.
    """
    # Initialize the database session
    db_session = init_db()

    # Create an instance of the ProvitasCore class
    provitas_core = ProvitasCore(db_session=db_session)

    # Record an action
    action_result = provitas_core.record_action(
        node_id="test_node_1",
        accuracy=0.8,
        statement="This is a test statement"
    )
    print("Action recorded:", action_result)

    # Get the network status
    network_status = provitas_core.get_network_status()
    print("\nNetwork Status:", network_status)

if __name__ == "__main__":
    main()