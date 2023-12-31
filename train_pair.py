import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
import csv
from data.tiled_pair_dataset import TiledPairDataset, SandedPairDataset, SandedPairTestDataset, ErodedPairDataset, ErodedPairTestDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_resnet(num_classes=5):
    resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)

    # Substitute the FC output layer
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet


class PairMatcher:

    def load_model(self):
        model_path = self.model_log_path / Path('model.pt')
        if model_path.exists():
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.net.eval()

        self.net.to(device=device)

    def save_model(self):
        torch.save(self.net.state_dict(), self.model_log_path / Path('model.pt'))

    def __init__(self, model_log_path: Path, num_classes=5):
        self.net = get_resnet(num_classes=num_classes)
        self.net = self.net.to(device=device)
        self.model_log_path = model_log_path
        self.model_log_path.mkdir(exist_ok=True, parents=True)
        self.load_model()
        self.criterion = None
        self.optimizer = None
        self.max_per_epoch = 500

    def train_epoch(self, train_dataloader):
        self.net.train()
        loss_sum = 0
        total_examples = 0
        for data in train_dataloader:
            inputs, labels = data
            # flatten
            inputs = inputs.view([-1] + list(inputs.shape[2:]))
            labels = labels.flatten()
            self.optimizer.zero_grad()
            prediction = self.net(inputs.to(device=device))
            loss = self.criterion(prediction, labels.to(device=device))
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()

            total_examples += inputs.shape[0]
            if total_examples > self.max_per_epoch:
                break

        return loss_sum / total_examples

    def test_epoch(self, test_dataloader):
        self.net.eval()
        loss_sum = 0
        correct = 0
        total = 0

        total_not_connected = 0
        correct_not_connected = 0

        total_connected = 0
        correct_connected = 0
        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)
                inputs = inputs.view([-1] + list(inputs.shape[2:]))
                labels = labels.flatten()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss_sum += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels.detach().cpu().tolist(), predicted.detach().cpu().tolist()):
                    if label == 0:
                        total_not_connected += 1
                        if prediction == 0:
                            correct_not_connected += 1

                    else:
                        total_connected += 1
                        if prediction == label:
                            correct_connected += 1

        return loss_sum / len(test_dataloader), correct / total, correct_not_connected / total_not_connected, correct_connected / total_connected


    def setup_training(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def get_data(self):
        train_dataloader = DataLoader(
            ErodedPairDataset(Path('./data/images/train')),
            batch_size=4,
            shuffle=True,
            num_workers=16
        )

        test_dataloader = DataLoader(
            ErodedPairTestDataset(Path('./data/images/high_detail_test')),
            batch_size=4,
            shuffle=True,
        )

        return train_dataloader, test_dataloader

    def train(self, epochs: int):
        self.setup_training()
        train_dataloader, test_dataloader = self.get_data()

        with open(self.model_log_path / Path('trace.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'test_loss', 'test_accuracy', 'not_connected_accuracy', 'connected_accuracy'])
            writer.writeheader()
            for epoch in tqdm(range(epochs)):
                train_loss = self.train_epoch(train_dataloader=train_dataloader)
                test_loss, test_accuracy, not_connected_accuracy, connected_accuracy = self.test_epoch(test_dataloader=test_dataloader)
                writer.writerow({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'not_connected_accuracy': not_connected_accuracy,
                    'connected_accuracy': connected_accuracy,
                })

        self.save_model()

    def infer(self, tiled_image: torch.Tensor):
        with torch.no_grad():
            net_output = self.net(tiled_image)
            return torch.nn.functional.softmax(net_output, dim=1)


if __name__ == '__main__':
    PairMatcher(model_log_path=Path('./models/eroded_pair_dataset_full')).train(epochs=200)



