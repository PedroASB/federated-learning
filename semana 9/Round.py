class Round:
    def __init__(self, network_model):
        self.network_model = network_model
        self.clients = None
        self.sender_clients = None
        self.rb_allocation = None

    def set_clients(self, clients):
        self.clients = clients

    def set_sender_clients(self, sender_clients):
        self.sender_clients = sender_clients
    
    def set_rb_allocation(self, rb_allocation):
        self.rb_allocation = rb_allocation

    def get_round_final_energy(self):
        assert self.clients is not None and self.rb_allocation is not None and self.sender_clients is not None

        return self.network_model.calculate_final_total_energy(self.clients, self.sender_clients, self.rb_allocation)