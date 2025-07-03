import torch
import torch.nn as nn
import timm
import numpy as np
from timm.models.vision_transformer import Block
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch



class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class ViGBLiF(nn.Module):
    def __init__(
        self,
        gcn_channels1: int = 64,
        gcn_channels2: int = 64,
        emb_dim: int = 128,
        k_neighboor: int = 12,
        fcc_1: int = 512,
        fcc_2: int = 64,
    ):
        super(ViGBLiF, self).__init__()

        # Initialize Vision Transformer (ViT)
        self.vit = timm.create_model("vit_base_patch8_224", pretrained=True)
        self.save_output = SaveOutput()

        # Freeze all layers of the Vision Transformer
        for param in self.vit.parameters():
            param.requires_grad = False

        #Register hook to save outputs of the Block modules in ViT
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.edge_index = self.get_edge_index((15, 3), k_neighboor)
        self.emb_dim = emb_dim

        self.conv = nn.Conv2d(768, self.emb_dim, 1, 1, 0)

        self.gconv1 = GCNConv(self.emb_dim * 28 * 28, gcn_channels1)
        self.gconv2 = GCNConv(gcn_channels1, gcn_channels2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gcn_channels2, fcc_1),
            nn.ELU(),
            nn.Linear(fcc_1, fcc_2),
            nn.Linear(fcc_2, 1),
        )

    def get_edge_index(self, shape: tuple, k: int) -> torch.Tensor:
        """
        Generates edge indices for graph connectivity based on k-nearest neighbors.

        Args:
            shape (tuple): Shape of the grid for nodes (height, width).
            k (int): Number of nearest neighbors to connect each node.

        Returns:
            torch.Tensor: Edge index tensor for PyTorch Geometric.
        """

        edges = []
        height, width = shape
        total_nodes = height * width
        node_positions = []

        # Create a list of (row, col) positions for each node
        for row in range(height):
            for col in range(width):
                node_positions.append((row, col))

        node_positions = np.array(node_positions)

        def euclidean_distance(node1: int, node2: int) -> float:
            """Computes the Euclidean distance between two nodes."""
            return np.linalg.norm(node_positions[node1] - node_positions[node2])

        # Connect each node to the k closest nodes
        for i in range(total_nodes):
            distances = []

            # Calculate distance from node i to all other nodes
            for j in range(total_nodes):
                if i != j:
                    distance = euclidean_distance(i, j)
                    distances.append((j, distance))

            # Sort distances in ascending order to get closest nodes
            distances.sort(key=lambda k: k[1])
            closest_nodes = distances[:k]

            # Create edges with the closest k nodes
            for node, _ in closest_nodes:
                edges.append([i, node])
                edges.append([node, i])

        # Convert to tensor for PyTorch Geometric
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index

    def forward(self, x, device):

        batch_size = x.shape[0]

        #Patch image
        x = (
            x.unfold(2, 224, 224)
            .unfold(3, 224, 112)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch_size, -1, 3, 224, 224)
        )
        x = x.view(batch_size * 15 * 3, 3, 224, 224)

        x = self.vit(x)
        x = self.save_output.outputs[9][:, 1:]
        self.save_output.outputs.clear()

        # Transform ViT output for GCN input
        x_graph = x.view(batch_size * 15 * 3, 28, 28, 768).permute(0, 3, 1, 2)
        x_graph = self.conv(x_graph)
        x_graph = x_graph.reshape(batch_size, 15 * 3, 28 * 28 * self.emb_dim)

        data_list = []
        for b in range(batch_size):
            node_features = x_graph[b].view(-1, x_graph[b].shape[-1])
            data = Data(x=node_features, edge_index=self.edge_index).to(device)
            data_list.append(data)


        batch_data = Batch.from_data_list(data_list)
        x_gcnn = self.gconv1(batch_data.x, batch_data.edge_index)
        x_gcnn = self.activation(x_gcnn)
        x_gcnn = self.gconv2(x_gcnn, batch_data.edge_index)
        x_gcnn = self.activation(x_gcnn)
        x_gcnn = global_mean_pool(x_gcnn, batch_data.batch)


        x = self.regression(x_gcnn)

        return x



if __name__ == "__main__":
    net = ViGBLiF().cuda()
    from thop import profile

    input1 = torch.randn(1, 3, 3360, 512).cuda()
    flops, params = profile(net, inputs=(input1,))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
