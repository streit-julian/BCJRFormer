import logging
from abc import ABC, abstractmethod
from typing import Tuple

import torch

from bcjrformer.codes.linear_code import LinearCode
from bcjrformer.codes.marker_code import MarkerCode
from bcjrformer.configs.code_config import (
    ConvolutionalCodeConfig,
    InnerCodeConfig,
    LinearCodeConfig,
    MarkerCodeConfig,
)
from bcjrformer.configs.evaluation_config import EvaluationConfig
from bcjrformer.configs.model_config import (
    IDSChannelConfig,
    ModelConfig,
)
from bcjrformer.configs.specific_model_config import (
    CombinedConvBCJRFormerModel,
    ConvBCJRFormerModel,
    EcctOnBCJRFormerE2EModel,
    SpecificModelConfig,
    SpecificModelIdentifier,
    specific_model_instance_to_identifier,
)
from bcjrformer.datasets.bcjrformer_dataset import (
    BCJRFormerConvCombinedDataset,
    BCJRFormerConvDataset,
    BCJRFormerConvStateDataset,
    BCJRFormerMarkerDataset,
)
from bcjrformer.datasets.ids_ecct_outer_dataset import (
    IdsConvEcctOuterDataset,
    IdsMarkerEcctOuterDataset,
)
from bcjrformer.trainers.bcjrformer_comb_cc_trainer import BCJRFormerCombConvTrainer
from bcjrformer.trainers.bcjrformer_marker_trainer import BCJRFormerMarkerTrainer
from bcjrformer.trainers.bcjrformer_trainer import BCJRFormerTrainer
from bcjrformer.trainers.convbcjrformer_trainer import ConvBCJRFormerTrainer
from bcjrformer.trainers.ecct_marker_bcjrformer_separate_e2e_trainer import (
    ECCTMarkerBCJRFormerSeparateE2ETrainer,
)
from bcjrformer.trainers.ids_ecct_outer_trainer import IdsEcctOuterTrainer
from bcjrformer.trainers.nb_bcjrformer_marker_trainer import NBBCJRFormerMarkerTrainer
from bcjrformer.codes.convolutional_code import (
    LinearConvolutionalCode,
    MarkerConvolutionalCode,
)
from bcjrformer.utils import evaluation_config_to_model_config

Trainer = (
    IdsEcctOuterTrainer
    | ECCTMarkerBCJRFormerSeparateE2ETrainer
    | BCJRFormerTrainer
    | BCJRFormerMarkerTrainer
    | NBBCJRFormerMarkerTrainer
    | BCJRFormerCombConvTrainer
    | ConvBCJRFormerTrainer
)
Dataset = (
    IdsConvEcctOuterDataset
    | IdsMarkerEcctOuterDataset
    | BCJRFormerConvDataset
    | BCJRFormerMarkerDataset
    | BCJRFormerConvStateDataset
    | BCJRFormerConvCombinedDataset
)


def get_linear_code_from_config(linear_code_config: LinearCodeConfig) -> LinearCode:
    return LinearCode.from_code_args(
        linear_code_config.code_type,
        linear_code_config.n,
        linear_code_config.k,
        linear_code_config.q,
        linear_code_config.file_extension,
        custom_file_name=linear_code_config.custom_file_name,
        random_weights=linear_code_config.random_weights,
    )


class BaseModelBuilder(ABC):
    @abstractmethod
    def build(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        code: LinearCode,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        """
        Creates and returns the train dataset, test dataset, and trainer.
        """
        pass

    def build_for_evaluation(
        self,
        evaluation_config: EvaluationConfig,
        channel_config: IDSChannelConfig,
        linear_code_config: LinearCodeConfig,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        """
        Creates and returns the test dataset and trainer for evaluation.
        """
        model_config = evaluation_config_to_model_config(evaluation_config)

        code = get_linear_code_from_config(linear_code_config)

        return self.build(
            model_config, channel_config, code, inner_code_config, device, logger
        )

    def build_for_train(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        linear_code_config: LinearCodeConfig,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        code = get_linear_code_from_config(linear_code_config)

        return self.build(
            model_config, channel_config, code, inner_code_config, device, logger
        )


class BCJRFormerBuilder(BaseModelBuilder):
    def build(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        code: LinearCode,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        # Ensure that the channel is of the expected type:

        if inner_code_config is None:
            raise ValueError("Inner code config must be provided for BCJRFormer")

        if isinstance(inner_code_config, MarkerCodeConfig):
            # Create the marker code
            marker_code = MarkerCode(
                marker=inner_code_config.marker,
                N_c=inner_code_config.N_c,
                p=inner_code_config.p,
                T=code.n,
            )
            # Create train & test datasets
            train_dataset = BCJRFormerMarkerDataset(
                linear_code=code,
                marker_code=marker_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.batch_size,
                batches_per_epoch=model_config.batches_per_epoch,
                compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
                std_mult=model_config.inner_model_config.delta_std_multiplier,
                n_sequence_min=model_config.inner_model_config.n_sequence_min,
                n_sequence_max=model_config.inner_model_config.n_sequence_max,
            )
            test_dataset = BCJRFormerMarkerDataset(
                linear_code=code,
                marker_code=marker_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.test_batch_size,
                batches_per_epoch=model_config.test_batches_per_epoch,
                compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
                std_mult=model_config.inner_model_config.delta_std_multiplier,
                n_sequence_min=model_config.inner_model_config.n_sequence_min,
                n_sequence_max=model_config.inner_model_config.n_sequence_max,
            )

            # Select trainer based on marker_code parameters.
            if marker_code.q == 2:
                trainer = BCJRFormerMarkerTrainer(
                    model_config,
                    test_dataset.window_block_dimension,
                    marker_code,
                    device,
                    logger,
                )
            else:
                trainer = NBBCJRFormerMarkerTrainer(
                    model_config,
                    test_dataset.window_block_dimension,
                    marker_code,
                    device,
                    marker_code.q,
                    logger,
                )
        elif isinstance(inner_code_config, ConvolutionalCodeConfig):
            conv_code = LinearConvolutionalCode(
                inner_code_config.k,
                inner_code_config.g,
                inner_code_config.p,
                code.n,
                random_offset=False,
            )

            train_dataset = BCJRFormerConvDataset(
                linear_code=code,
                conv_code=conv_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.batch_size,
                batches_per_epoch=model_config.batches_per_epoch,
                compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
                std_mult=model_config.inner_model_config.delta_std_multiplier,
                n_sequence_min=model_config.inner_model_config.n_sequence_min,
                n_sequence_max=model_config.inner_model_config.n_sequence_max,
            )
            test_dataset = BCJRFormerConvDataset(
                linear_code=code,
                conv_code=conv_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.test_batch_size,
                batches_per_epoch=model_config.test_batches_per_epoch,
                compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
                std_mult=model_config.inner_model_config.delta_std_multiplier,
                n_sequence_min=model_config.inner_model_config.n_sequence_min,
                n_sequence_max=model_config.inner_model_config.n_sequence_max,
            )

            trainer = BCJRFormerTrainer(
                model_config,
                test_dataset.window_block_dimension,
                test_dataset.trellis.encoded_length,
                channel_config,
                device,
                logger,
            )
        else:
            raise ValueError(
                "BCJRFormer requires either MarkerCodeConfig or ConvolutionalCodeConfig as inner code config"
            )

        return train_dataset, test_dataset, trainer


class IdsEcctOuterBuilder(BaseModelBuilder):
    def build(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        code: LinearCode,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        if inner_code_config is None:
            raise ValueError("Inner code config must be provided for IDS_ECCT_OUTER")

        if channel_config.fixed_del_count is not None:
            raise ValueError("CC_FB_ECCT_OUTER does not support fixed deletion count")

        if isinstance(inner_code_config, ConvolutionalCodeConfig):
            conv_code = LinearConvolutionalCode(
                inner_code_config.k,
                inner_code_config.g,
                inner_code_config.p,
                code.n,
                random_offset=True,
            )
            train_dataset = IdsConvEcctOuterDataset(
                code=code,
                conv_code=conv_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.batch_size,
                batches_per_epoch=model_config.batches_per_epoch,
                use_zero_cw=False,
            )
            test_dataset = IdsConvEcctOuterDataset(
                code=code,
                conv_code=conv_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.test_batch_size,
                batches_per_epoch=model_config.test_batches_per_epoch,
                use_zero_cw=False,
            )
        elif isinstance(inner_code_config, MarkerCodeConfig):
            marker_code = MarkerCode(
                marker=inner_code_config.marker,
                N_c=inner_code_config.N_c,
                p=inner_code_config.p,
                T=code.n,
            )
            conv_code = MarkerConvolutionalCode(
                T=marker_code.encoded_length, p=marker_code.p, random_offset=False
            )
            train_dataset = IdsMarkerEcctOuterDataset(
                code=code,
                marker_code=marker_code,
                conv_code=conv_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.batch_size,
                batches_per_epoch=model_config.batches_per_epoch,
                use_zero_cw=False,
            )
            test_dataset = IdsMarkerEcctOuterDataset(
                code=code,
                marker_code=marker_code,
                conv_code=conv_code,
                p_i=channel_config.p_i,
                p_d=channel_config.p_d,
                p_s=channel_config.p_s,
                batch_size=model_config.test_batch_size,
                batches_per_epoch=model_config.test_batches_per_epoch,
                use_zero_cw=False,
            )
        else:
            raise ValueError("Invalid inner code config")
        trainer = IdsEcctOuterTrainer(model_config, code, device, logger)
        return train_dataset, test_dataset, trainer


class ECCTOnBcjrFormerSeparateE2EBuilder(BaseModelBuilder):
    def build(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        code: LinearCode,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:

        if not isinstance(inner_code_config, MarkerCodeConfig):
            raise NotImplementedError(
                "IDS_ECCT_OUTER_BCJRFORMER_E2E requires MarkerCodeConfig as inner code config for now"
            )

        if not isinstance(model_config.specific_model, EcctOnBCJRFormerE2EModel):
            raise ValueError("Unexpected inner model config. This should not happen.")

        ecct_on_bcjrformer_e2e_config = model_config.specific_model
        marker_code = MarkerCode(
            marker=inner_code_config.marker,
            N_c=inner_code_config.N_c,
            p=inner_code_config.p,
            T=code.n,
        )
        train_dataset = BCJRFormerMarkerDataset(
            linear_code=code,
            marker_code=marker_code,
            p_i=channel_config.p_i,
            p_d=channel_config.p_d,
            p_s=channel_config.p_s,
            batch_size=model_config.batch_size,
            batches_per_epoch=model_config.batches_per_epoch,
            compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
            std_mult=model_config.inner_model_config.delta_std_multiplier,
            n_sequence_min=model_config.inner_model_config.n_sequence_min,
            n_sequence_max=model_config.inner_model_config.n_sequence_max,
        )
        test_dataset = BCJRFormerMarkerDataset(
            linear_code=code,
            marker_code=marker_code,
            p_i=channel_config.p_i,
            p_d=channel_config.p_d,
            p_s=channel_config.p_s,
            batch_size=model_config.test_batch_size,
            batches_per_epoch=model_config.test_batches_per_epoch,
            compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
            std_mult=model_config.inner_model_config.delta_std_multiplier,
            n_sequence_min=model_config.inner_model_config.n_sequence_min,
            n_sequence_max=model_config.inner_model_config.n_sequence_max,
        )
        trainer = ECCTMarkerBCJRFormerSeparateE2ETrainer(
            model_config,
            ecct_on_bcjrformer_e2e_config,
            test_dataset.window_block_dimension,
            code,
            marker_code,
            channel_config,
            device,
            logger,
        )
        return train_dataset, test_dataset, trainer


class CombinedConvBcjrFormerBuilder(BaseModelBuilder):
    def build(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        code: LinearCode,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        if not isinstance(inner_code_config, ConvolutionalCodeConfig):
            raise ValueError(
                "CombinedConvBcjrFormer requires ConvolutionalCodeConfig as inner code config"
            )

        if not isinstance(model_config.specific_model, CombinedConvBCJRFormerModel):
            raise ValueError("Unexpected inner model config. This should not happen.")

        combined_conv_bcjrformer_config = model_config.specific_model

        conv_code = LinearConvolutionalCode(
            inner_code_config.k,
            inner_code_config.g,
            inner_code_config.p,
            code.n,
            random_offset=inner_code_config.random_offset,
        )
        train_dataset = BCJRFormerConvCombinedDataset(
            linear_code=code,
            conv_code=conv_code,
            p_i=channel_config.p_i,
            p_d=channel_config.p_d,
            p_s=channel_config.p_s,
            batch_size=model_config.batch_size,
            batches_per_epoch=model_config.batches_per_epoch,
            compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
            std_mult=model_config.inner_model_config.delta_std_multiplier,
        )
        test_dataset = BCJRFormerConvCombinedDataset(
            linear_code=code,
            conv_code=conv_code,
            p_i=channel_config.p_i,
            p_d=channel_config.p_d,
            p_s=channel_config.p_s,
            batch_size=model_config.test_batch_size,
            batches_per_epoch=model_config.test_batches_per_epoch,
            compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
            std_mult=model_config.inner_model_config.delta_std_multiplier,
        )

        trainer = BCJRFormerCombConvTrainer(
            model_config,
            combined_conv_bcjrformer_config,
            conv_code,
            test_dataset.bit_window_block_dimension,
            test_dataset.state_window_block_dimension,
            test_dataset.bit_trellis.T,
            test_dataset.state_trellis.T,
            channel_config,
            device,
            logger,
        )
        return train_dataset, test_dataset, trainer


class ConvBcjrFormerBuilder(BaseModelBuilder):
    def build(
        self,
        model_config: ModelConfig,
        channel_config: IDSChannelConfig,
        code: LinearCode,
        inner_code_config: InnerCodeConfig | None,
        device: torch.device,
        logger: logging.Logger,
    ) -> Tuple[Dataset, Dataset, Trainer]:
        if not isinstance(inner_code_config, ConvolutionalCodeConfig):
            raise ValueError(
                "ConvBcjrFormer requires ConvolutionalCodeConfig as inner code config"
            )

        if not isinstance(model_config.specific_model, ConvBCJRFormerModel):
            raise ValueError("Unexpected inner model config. This should not happen.")

        conv_bcjrformer_config = model_config.specific_model

        conv_code = LinearConvolutionalCode(
            inner_code_config.k,
            inner_code_config.g,
            inner_code_config.p,
            code.n,
            random_offset=inner_code_config.random_offset,
        )
        train_dataset = BCJRFormerConvCombinedDataset(
            linear_code=code,
            conv_code=conv_code,
            p_i=channel_config.p_i,
            p_d=channel_config.p_d,
            p_s=channel_config.p_s,
            batch_size=model_config.batch_size,
            batches_per_epoch=model_config.batches_per_epoch,
            compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
            std_mult=model_config.inner_model_config.delta_std_multiplier,
        )
        test_dataset = BCJRFormerConvCombinedDataset(
            linear_code=code,
            conv_code=conv_code,
            p_i=channel_config.p_i,
            p_d=channel_config.p_d,
            p_s=channel_config.p_s,
            batch_size=model_config.test_batch_size,
            batches_per_epoch=model_config.test_batches_per_epoch,
            compare_bcjr=bool(model_config.inner_model_config.compare_bcjr),
            std_mult=model_config.inner_model_config.delta_std_multiplier,
        )

        trainer = ConvBCJRFormerTrainer(
            model_config,
            conv_bcjrformer_config,
            conv_code,
            test_dataset.bit_window_block_dimension,
            test_dataset.state_window_block_dimension,
            test_dataset.bit_trellis.T,
            test_dataset.state_trellis.T,
            device,
            logger,
        )
        return train_dataset, test_dataset, trainer


def model_builder_factory(
    specific_model_config: SpecificModelConfig,
) -> BaseModelBuilder:
    identifier = specific_model_instance_to_identifier(specific_model_config)
    builders = {
        SpecificModelIdentifier.BCJRFORMER: BCJRFormerBuilder,
        SpecificModelIdentifier.IDS_ECCT_OUTER: IdsEcctOuterBuilder,
        SpecificModelIdentifier.ECCT_ON_BCJRFORMER_SEP_E2E: ECCTOnBcjrFormerSeparateE2EBuilder,
        SpecificModelIdentifier.COMBINED_CONV_BCJRFORMER: CombinedConvBcjrFormerBuilder,
        SpecificModelIdentifier.CONV_BCJRFORMER: ConvBcjrFormerBuilder,
    }
    try:
        return builders[identifier]()
    except KeyError as e:
        raise ValueError("Invalid model config") from e


def get_datasets_and_trainer(
    model_config: ModelConfig,
    specific_model_config: SpecificModelConfig,
    channel_config: IDSChannelConfig,
    linear_code_config: LinearCodeConfig,
    inner_code_config: InnerCodeConfig | None,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[Dataset, Dataset, Trainer]:
    """
    Generates the datasets (train & test) and a trainer based on the model configuration.
    """
    linear_code = LinearCode.from_code_args(
        linear_code_config.code_type,
        linear_code_config.n,
        linear_code_config.k,
        linear_code_config.q,
        linear_code_config.file_extension,
        custom_file_name=linear_code_config.custom_file_name,
        random_weights=linear_code_config.random_weights,
    )
    builder = model_builder_factory(specific_model_config)
    return builder.build(
        model_config, channel_config, linear_code, inner_code_config, device, logger
    )


##################################################################
