import torch
from torch import nn, optim
from unittest.mock import MagicMock, patch

from diffusionlab.models import DiffusionModel
from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.schedulers import Scheduler
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.losses import SamplewiseDiffusionLoss


class TestDiffusionModel:
    def test_initialization(self):
        """Test basic initialization of DiffusionModel."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)
        train_scheduler = MagicMock(spec=Scheduler)
        vector_field_type = VectorFieldType.EPS
        optimizer = MagicMock(spec=optim.Optimizer)
        lr_scheduler = MagicMock(spec=optim.lr_scheduler.LRScheduler)
        batchwise_metrics = {}
        batchfree_metrics = {}
        train_ts_hparams = {"t_min": 0.001, "t_max": 0.99, "L": 10}
        t_loss_weights = lambda t: torch.ones_like(t)
        t_loss_probs = lambda t: torch.ones_like(t) / t.shape[0]
        N_noise_draws_per_sample = 2

        # Mock the precompute_train_schedule method to avoid actual computation
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=train_scheduler,
                vector_field_type=vector_field_type,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                batchwise_metrics=batchwise_metrics,
                batchfree_metrics=batchfree_metrics,
                train_ts_hparams=train_ts_hparams,
                t_loss_weights=t_loss_weights,
                t_loss_probs=t_loss_probs,
                N_noise_draws_per_sample=N_noise_draws_per_sample,
            )

        # Check that attributes are set correctly
        assert model.net is net
        assert model.diffusion_process is diffusion_process
        assert model.train_scheduler is train_scheduler
        assert model.vector_field_type is vector_field_type
        assert model.optimizer is optimizer
        assert model.lr_scheduler is lr_scheduler
        assert isinstance(model.batchwise_metrics, nn.ModuleDict)
        assert isinstance(model.batchfree_metrics, nn.ModuleDict)
        assert model.t_loss_weights is t_loss_weights
        assert model.t_loss_probs is t_loss_probs
        assert model.N_noise_draws_per_sample == N_noise_draws_per_sample
        assert isinstance(model.samplewise_loss, SamplewiseDiffusionLoss)

    def test_precompute_train_schedule(self):
        """Test precomputing training schedule."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)
        train_scheduler = MagicMock(spec=Scheduler)
        vector_field_type = VectorFieldType.EPS
        optimizer = MagicMock(spec=optim.Optimizer)
        lr_scheduler = MagicMock(spec=optim.lr_scheduler.LRScheduler)
        train_ts_hparams = {"t_min": 0.001, "t_max": 0.99, "L": 10}

        # Create mock time steps
        mock_ts = torch.linspace(0.001, 0.99, 10)
        train_scheduler.get_ts.return_value = mock_ts

        # Create weight and probability functions
        t_loss_weights = lambda t: torch.ones_like(t)
        t_loss_probs = lambda t: torch.ones_like(t) / t.shape[0]

        # Initialize model with mocked precompute_train_schedule
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=train_scheduler,
                vector_field_type=vector_field_type,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams=train_ts_hparams,
                t_loss_weights=t_loss_weights,
                t_loss_probs=t_loss_probs,
                N_noise_draws_per_sample=2,
            )

        # Call the method directly
        model.precompute_train_schedule(train_ts_hparams)

        # Check that scheduler was called with correct parameters
        train_scheduler.get_ts.assert_called_once_with(**train_ts_hparams)

        # Check that buffers were updated
        assert torch.allclose(model.train_ts, mock_ts)
        assert torch.allclose(model.train_ts_loss_weights, t_loss_weights(mock_ts))
        assert torch.allclose(model.train_ts_loss_probs, t_loss_probs(mock_ts))

    def test_forward(self):
        """Test forward method of DiffusionModel."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)
        train_scheduler = MagicMock(spec=Scheduler)

        # Set up the mock network to return a specific value
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        t = torch.ones(batch_size) * 0.5
        expected_output = torch.randn(batch_size, data_dim)
        net.return_value = expected_output

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=train_scheduler,
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Call forward
        output = model.forward(x, t)

        # Check that net was called with correct arguments
        net.assert_called_once_with(x, t)

        # Check that output matches expected
        assert torch.equal(output, expected_output)

    def test_configure_optimizers(self):
        """Test configure_optimizers method."""
        # Create mock objects
        optimizer = MagicMock(spec=optim.Optimizer)
        lr_scheduler = MagicMock(spec=optim.lr_scheduler.LRScheduler)

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=MagicMock(spec=nn.Module),
                diffusion_process=MagicMock(spec=DiffusionProcess),
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Call configure_optimizers
        result = model.configure_optimizers()

        # Check that the result is a dictionary with the correct keys and values
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["optimizer"] is optimizer
        assert result["lr_scheduler"] is lr_scheduler

    def test_loss(self):
        """Test loss computation."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Set up input data
        batch_size = 3
        data_dim = 2
        x = torch.randn(batch_size, data_dim)
        t = torch.ones(batch_size) * 0.5
        sample_weights = torch.ones(batch_size)

        # Create a mock for SamplewiseDiffusionLoss
        mock_loss = MagicMock()
        mock_loss.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Create a proper scheduler that returns a tensor
        train_scheduler = MagicMock(spec=Scheduler)
        train_scheduler.get_ts.return_value = torch.linspace(0.001, 0.99, 10)

        # Initialize model with patched SamplewiseDiffusionLoss and precompute_train_schedule
        with (
            patch(
                "diffusionlab.models.SamplewiseDiffusionLoss", return_value=mock_loss
            ),
            patch.object(DiffusionModel, "precompute_train_schedule"),
        ):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=train_scheduler,
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Manually set up the samplewise_loss attribute
        model.samplewise_loss = mock_loss

        # Create a consistent noise tensor for testing
        noise = torch.randn(batch_size * 2, data_dim)

        # Mock torch.randn_like to return our controlled noise
        with patch("torch.randn_like", return_value=noise):
            # Call loss method
            result = model.loss(x, t, sample_weights)

        # Verify the result is the mean of the loss values
        expected_mean = torch.mean(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        assert torch.isclose(result, expected_mean)

    def test_aggregate_loss(self):
        """Test aggregate_loss method."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Set up input data
        batch_size = 3
        data_dim = 2
        x = torch.randn(batch_size, data_dim)

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Set up train_ts and related buffers
        model.register_buffer("train_ts", torch.linspace(0.001, 0.99, 10))
        model.register_buffer("train_ts_loss_weights", torch.ones(10))
        model.register_buffer("train_ts_loss_probs", torch.ones(10) / 10)

        # Create a mock for the loss method
        mock_loss = MagicMock(return_value=torch.tensor(0.5))

        # Use the mock to replace the loss method
        with patch.object(model, "loss", mock_loss):
            # Call aggregate_loss
            result = model.aggregate_loss(x)

            # Verify loss was called (at least once)
            assert mock_loss.call_count > 0

            # Verify the result matches our expected value
            assert torch.equal(result, torch.tensor(0.5))

    def test_training_step(self):
        """Test training_step method."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Set up input data
        batch_size = 3
        data_dim = 2
        x = torch.randn(batch_size, data_dim)
        metadata = torch.zeros(batch_size)
        batch = (x, metadata)
        batch_idx = 0

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Create mocks for the methods we want to test
        mock_aggregate_loss = MagicMock(return_value=torch.tensor(0.5))
        mock_log = MagicMock()

        # Patch the methods with our mocks
        with (
            patch.object(model, "aggregate_loss", mock_aggregate_loss),
            patch.object(model, "log", mock_log),
        ):
            # Call training_step
            result = model.training_step(batch, batch_idx)

            # Verify aggregate_loss was called with x
            mock_aggregate_loss.assert_called_once()
            args, _ = mock_aggregate_loss.call_args
            assert torch.equal(args[0], x)

            # Verify log was called with the loss
            mock_log.assert_called_once()
            args, _ = mock_log.call_args
            assert args[0] == "train_loss"
            assert torch.equal(args[1], torch.tensor(0.5))

            # Verify the result is the loss
            assert torch.equal(result, torch.tensor(0.5))

    def test_validation_step_no_metrics(self):
        """Test validation_step method with no metrics."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Set up input data
        batch_size = 3
        data_dim = 2
        x = torch.randn(batch_size, data_dim)
        metadata = torch.zeros(batch_size)
        batch = (x, metadata)
        batch_idx = 0

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Create mocks for the methods we want to test
        mock_aggregate_loss = MagicMock(return_value=torch.tensor(0.5))
        mock_log_dict = MagicMock()

        # Patch the methods with our mocks
        with (
            patch.object(model, "aggregate_loss", mock_aggregate_loss),
            patch.object(model, "log_dict", mock_log_dict),
        ):
            # Call validation_step
            result = model.validation_step(batch, batch_idx)

            # Verify aggregate_loss was called with x
            mock_aggregate_loss.assert_called_once()
            args, _ = mock_aggregate_loss.call_args
            assert torch.equal(args[0], x)

            # Verify log_dict was called with the loss
            mock_log_dict.assert_called_once()
            args, _ = mock_log_dict.call_args
            assert args[0] == {"val_loss": torch.tensor(0.5)}

            # Verify the result is the expected dictionary
            assert result == {"val_loss": torch.tensor(0.5)}

    def test_validation_step_with_metrics(self):
        """Test validation_step method with metrics."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Set up input data
        batch_size = 3
        data_dim = 2
        x = torch.randn(batch_size, data_dim)
        metadata = torch.zeros(batch_size)
        batch = (x, metadata)
        batch_idx = 0

        # Create a counter to track metric calls
        metric_calls = {"metric1": 0, "metric2": 0}

        # Create mock metrics that are nn.Module subclasses
        class MockMetric1(nn.Module):
            def forward(self, x, metadata, model):
                metric_calls["metric1"] += 1
                return {"value1": torch.tensor(0.1), "value2": torch.tensor(0.2)}

        class MockMetric2(nn.Module):
            def forward(self, x, metadata, model):
                metric_calls["metric2"] += 1
                return {"value3": torch.tensor(0.3)}

        # Initialize model with metrics
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={"metric1": MockMetric1(), "metric2": MockMetric2()},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Create mocks for the methods we want to test
        mock_aggregate_loss = MagicMock(return_value=torch.tensor(0.5))
        mock_log_dict = MagicMock()

        # Patch the methods with our mocks
        with (
            patch.object(model, "aggregate_loss", mock_aggregate_loss),
            patch.object(model, "log_dict", mock_log_dict),
        ):
            # Call validation_step
            result = model.validation_step(batch, batch_idx)

            # Verify aggregate_loss was called with x
            mock_aggregate_loss.assert_called_once()
            args, _ = mock_aggregate_loss.call_args
            assert torch.equal(args[0], x)

            # Verify metrics were called
            assert metric_calls["metric1"] > 0
            assert metric_calls["metric2"] > 0

            # Verify log_dict was called with the expected values
            mock_log_dict.assert_called_once()
            args, _ = mock_log_dict.call_args
            expected_dict = {
                "val_loss": torch.tensor(0.5),
                "metric1_value1": torch.tensor(0.1),
                "metric1_value2": torch.tensor(0.2),
                "metric2_value3": torch.tensor(0.3),
            }
            assert set(args[0].keys()) == set(expected_dict.keys())
            for key in expected_dict:
                assert torch.equal(args[0][key], expected_dict[key])

            # Verify the result has the expected keys and values
            assert set(result.keys()) == set(expected_dict.keys())
            for key in expected_dict:
                assert torch.equal(result[key], expected_dict[key])

    def test_on_validation_epoch_end_no_metrics(self):
        """Test on_validation_epoch_end method with no metrics."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Create a mock for log_dict
        mock_log_dict = MagicMock()

        # Patch log_dict with our mock
        with patch.object(model, "log_dict", mock_log_dict):
            # Call on_validation_epoch_end
            model.on_validation_epoch_end()

            # Verify log_dict was called with empty dict
            mock_log_dict.assert_called_once()
            args, kwargs = mock_log_dict.call_args
            assert args[0] == {}
            assert kwargs["on_step"] == model.LOG_ON_STEP
            assert kwargs["on_epoch"] == model.LOG_ON_EPOCH
            assert kwargs["prog_bar"] == model.LOG_ON_PROGRESS_BAR

    def test_on_validation_epoch_end_with_metrics(self):
        """Test on_validation_epoch_end method with metrics."""
        # Create mock objects
        net = MagicMock(spec=nn.Module)
        diffusion_process = MagicMock(spec=DiffusionProcess)

        # Create a counter to track metric calls
        metric_calls = {"metric1": 0, "metric2": 0}

        # Create mock metrics that are nn.Module subclasses
        class MockMetric1(nn.Module):
            def forward(self, model):
                metric_calls["metric1"] += 1
                return {"value1": torch.tensor(0.1), "value2": torch.tensor(0.2)}

        class MockMetric2(nn.Module):
            def forward(self, model):
                metric_calls["metric2"] += 1
                return {"value3": torch.tensor(0.3)}

        # Initialize model
        with patch.object(DiffusionModel, "precompute_train_schedule"):
            model = DiffusionModel(
                net=net,
                diffusion_process=diffusion_process,
                train_scheduler=MagicMock(spec=Scheduler),
                vector_field_type=VectorFieldType.EPS,
                optimizer=MagicMock(spec=optim.Optimizer),
                lr_scheduler=MagicMock(spec=optim.lr_scheduler.LRScheduler),
                batchwise_metrics={},
                batchfree_metrics={"metric1": MockMetric1(), "metric2": MockMetric2()},
                train_ts_hparams={"t_min": 0.001, "t_max": 0.99, "L": 10},
                t_loss_weights=lambda t: torch.ones_like(t),
                t_loss_probs=lambda t: torch.ones_like(t) / t.shape[0],
                N_noise_draws_per_sample=2,
            )

        # Create a mock for log_dict
        mock_log_dict = MagicMock()

        # Patch log_dict with our mock
        with patch.object(model, "log_dict", mock_log_dict):
            # Call on_validation_epoch_end
            model.on_validation_epoch_end()

            # Verify metrics were called
            assert metric_calls["metric1"] > 0
            assert metric_calls["metric2"] > 0

            # Verify log_dict was called with metric values
            mock_log_dict.assert_called_once()
            args, kwargs = mock_log_dict.call_args
            expected_dict = {
                "metric1_value1": torch.tensor(0.1),
                "metric1_value2": torch.tensor(0.2),
                "metric2_value3": torch.tensor(0.3),
            }
            assert set(args[0].keys()) == set(expected_dict.keys())
            for key in expected_dict:
                assert torch.equal(args[0][key], expected_dict[key])
            assert kwargs["on_step"] == model.LOG_ON_STEP
            assert kwargs["on_epoch"] == model.LOG_ON_EPOCH
            assert kwargs["prog_bar"] == model.LOG_ON_PROGRESS_BAR
