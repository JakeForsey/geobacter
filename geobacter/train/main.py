from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from sklearn.manifold import TSNE
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import trange

from geobacter.inference.networks.resnet import ResNetTriplet
from geobacter.inference.networks.resnet import ResNetEmbedding
from geobacter.train.loss import TripletLoss
from geobacter.inference.datasets.osm import OsmTileDataset
from geobacter.inference.datasets.osm import DENORMALIZE

BATCH_SIZE = 32
TRAIN_EPOCHS = 50
CACHE_DIR = Path("data/cache")
matplotlib.use('Agg')


def main():
    run_id = str(uuid4())

    print("Initialising embedding network.")
    embedding_model = ResNetEmbedding(16)
    embedding_model.cuda()

    print("Initialising triplet network.")
    triplet_model = ResNetTriplet(embedding_model)
    triplet_model.cuda()

    print("Initialising training dataset.")
    train_dataset = OsmTileDataset(
        extents_path=Path("data/extents/train.pickle"),
        cache_dir=CACHE_DIR
    )

    print("Initialising testing dataset.")
    test_dataset = OsmTileDataset(
        extents_path=Path("data/extents/test.pickle"),
        cache_dir=CACHE_DIR
    )

    print("Fetching image entropy values.")
    entropy_values = [train_dataset.anchor_entropy(i) for i in trange(len(train_dataset), desc="calculating entropy")]
    # shannon_entropy of 1 means all pixels the same, these images have minimal value. Highest
    # value I've seen was around 3.
    weights = [entropy - 0.9 for entropy in entropy_values]
    print("Initialising sampler.")
    sampler = WeightedRandomSampler(
        weights=weights,
        # Only the 50% most entropic images
        num_samples=len(train_dataset) // 2,
        # No duplicates in an epoch
        replacement=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=4,
        sampler=sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4
    )

    triplet_loss = TripletLoss(margin=1)

    optimizer = optim.Adam(
        embedding_model.parameters(),
        lr=1e-3
    )
    scheduler = ExponentialLR(optimizer, 0.95)

    def train_step(engine, batch):
        embedding_model.train()
        triplet_model.train()

        anchor, positive, negative = batch
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

        optimizer.zero_grad()

        anchor_embedding, positive_embedding, negative_embedding = triplet_model(anchor, positive, negative)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding).cuda()

        loss.backward()
        optimizer.step()
        scheduler.step()

        return {
            'loss': loss.item(),
        }

    trainer = Engine(train_step)

    tb_logger = TensorboardLogger(log_dir=f"tensorboard/{run_id}")
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training",
            output_transform=lambda out: out,
            metric_names='all'
        ),
        event_name=Events.ITERATION_COMPLETED
    )
    # Plot some images with corresponding entropy values
    for idx, entropy in enumerate(entropy_values):
        if idx > 20:
            break

        image, _, _ = train_dataset[idx]
        image = DENORMALIZE(image)
        tb_logger.writer.add_image(
            f"entropy_{entropy}",
            image,
            global_step=0
        )

    @trainer.on(Events.EPOCH_STARTED)
    def add_test_images(engine):
        for idx, sample in enumerate(test_loader):
            if idx > 9:
                break
            anchor, positive, negative = sample
            anchor = DENORMALIZE(anchor.squeeze())
            positive = DENORMALIZE(positive.squeeze())
            negative = DENORMALIZE(negative.squeeze())

            tb_logger.writer.add_image(
                f"test_image_{idx}",
                torch.cat([anchor, positive, negative], 2),
                global_step=engine.state.epoch
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def test(engine):
        embedding_model.eval()
        triplet_model.eval()

        with torch.no_grad():
            embeddings = []
            images = []
            loss_total = 0
            for idx, sample in enumerate(test_loader):
                anchor, positive, negative = sample
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

                anchor_embedding, positive_embedding, negative_embedding = triplet_model(anchor, positive, negative)
                loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
                loss_total += loss.item()

                # 300 is a good number of images to plot
                if len(embeddings) < 300:
                    embeddings.append(
                        anchor_embedding.squeeze().detach().cpu().numpy()
                    )
                    images.append(
                        DENORMALIZE(anchor.squeeze()).detach().cpu().numpy().transpose(1, 2, 0)
                    )

        fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(9, 9)
        ax = plt.subplot(111)

        embeddings = TSNE(n_components=2).fit_transform(embeddings)
        for embedding_idx, image in enumerate(images):
            offset_image = OffsetImage(image, zoom=.2)
            ab = AnnotationBbox(
                offset_image,
                embeddings[embedding_idx],
                xybox=(30.0, -30.0),
                xycoords='data',
                boxcoords="offset points",
                frameon=False
            )
            ax.add_artist(ab)

        plt.axis("off")
        plt.xlim((embeddings[:, 0].min(), embeddings[:, 0].max()))
        plt.ylim((embeddings[:, 1].min(), embeddings[:, 1].max()))
        plt.draw()

        tb_logger.writer.add_figure(
            f"test_embeddings",
            fig,
            global_step=engine.state.epoch
        )
        tb_logger.writer.add_scalar(
            f"test_loss",
            # Assumes test loader has batch size of 1
            loss_total / len(test_loader),
            global_step=engine.state.epoch
        )

    print("Initialising checkpoint handler.")
    checkpoint_handler = ModelCheckpoint(
        "checkpoints/", f"{triplet_model.__class__.__name__}-{train_dataset.__class__.__name__}-{run_id}",
        n_saved=10,
        require_empty=False
    )
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={
            'embedding': embedding_model,
        })

    print("Initialising timer.")
    timer = Timer(average=True)
    timer.attach(
        trainer,
        resume=Events.ITERATION_STARTED,
        step=Events.ITERATION_COMPLETED
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        print(
            "Epoch[{}] Iteration[{}] Duration[{}] Losses: {}".format(
                engine.state.epoch,
                engine.state.iteration,
                timer.value(),
                engine.state.output
            )
        )

    @trainer.on(Events.EPOCH_STARTED)
    def log_learning_rate(engine):
        tb_logger.writer.add_scalar(
            f"learning_rate",
            torch.tensor(scheduler.get_lr()),
            global_step=engine.state.epoch
        )

    print("Running trainer.")
    trainer.run(train_loader, max_epochs=TRAIN_EPOCHS)

    tb_logger.close()


if __name__ == "__main__":
    main()
