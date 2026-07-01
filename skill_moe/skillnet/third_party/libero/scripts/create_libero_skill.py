"""This is a standalone file for create a task in libero."""
import os
from pathlib import Path

import numpy as np

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    get_task_info,
    generate_bddl_from_task_info,
)


BDDL_OUTPUT_DIR = Path(os.environ.get("LIBERO_BDDL_OUTPUT_DIR", "generated_bddl"))


@register_mu(scene_type="kitchen")
class KitchenScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.25],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene2(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 3,
            "plate": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.20],
                region_name="akita_black_bowl_middle_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, 0.15],
                region_name="akita_black_bowl_front_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.05],
                region_name="akita_black_bowl_back_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            (
                "On",
                "akita_black_bowl_1",
                "kitchen_table_akita_black_bowl_front_init_region",
            ),
            (
                "On",
                "akita_black_bowl_2",
                "kitchen_table_akita_black_bowl_middle_init_region",
            ),
            (
                "On",
                "akita_black_bowl_3",
                "kitchen_table_akita_black_bowl_back_init_region",
            ),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene3(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
        }

        object_num_info = {"chefmate_8_frypan": 1, "moka_pot": 1}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.20],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, -0.25],
                region_name="frypan_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.0],
                region_name="moka_pot_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "chefmate_8_frypan_1", "kitchen_table_frypan_init_region"),
            ("On", "moka_pot_1", "kitchen_table_moka_pot_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene4(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "white_cabinet": 1,
            "wine_rack": 1,
        }

        object_num_info = {"akita_black_bowl": 1, "wine_bottle": 1}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.30],
                region_name="white_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.30],
                region_name="wine_rack_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.03, -0.05],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.05],
                region_name="wine_bottle_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "wine_bottle_1", "kitchen_table_wine_bottle_init_region"),
            ("On", "white_cabinet_1", "kitchen_table_white_cabinet_init_region"),
            ("On", "wine_rack_1", "kitchen_table_wine_rack_init_region"),
            ("Open", "white_cabinet_1_bottom_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene5(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "white_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
            "ketchup": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.30],
                region_name="white_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.03, -0.05],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.10],
                region_name="ketchup_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, -0.25],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "white_cabinet_1", "kitchen_table_white_cabinet_init_region"),
            ("On", "ketchup_1", "kitchen_table_ketchup_init_region"),
            ("Open", "white_cabinet_1_top_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene6(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "porcelain_mug": 1,
            "white_yellow_mug": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.35],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(0, 0),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="white_yellow_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.25],
                region_name="porcelain_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.25],
                region_name="porcelain_mug_front_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "porcelain_mug_1", "kitchen_table_porcelain_mug_init_region"),
            ("On", "white_yellow_mug_1", "kitchen_table_white_yellow_mug_init_region"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Open", "microwave_1"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene7(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "white_bowl": 1,
            "plate": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.25],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.10],
                region_name="plate_right_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "white_bowl_1", "microwave_1_top_side"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Close", "microwave_1"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene8(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
        }

        object_num_info = {"moka_pot": 2}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.20],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.25],
                region_name="moka_pot_right_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.05],
                region_name="moka_pot_left_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"),
            ("On", "moka_pot_2", "kitchen_table_moka_pot_left_init_region"),
            ("Turnon", "flat_stove_1"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene9(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
            "wooden_two_layer_shelf": 1,
        }

        object_num_info = {
            "white_bowl": 1,
            "chefmate_8_frypan": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.30],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, -0.25],
                region_name="wooden_two_layer_shelf_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.0],
                region_name="frypan_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.10],
                region_name="white_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "chefmate_8_frypan_1", "kitchen_table_frypan_init_region"),
            ("On", "white_bowl_1", "kitchen_table_white_bowl_init_region"),
            (
                "On",
                "wooden_two_layer_shelf_1",
                "kitchen_table_wooden_two_layer_shelf_init_region",
            ),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene10(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "butter": 2,
            "chocolate_pudding": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, 0.0],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, 0.20],
                region_name="butter_back_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.20],
                region_name="butter_front_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.05],
                region_name="chocolate_pudding_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "butter_1", "kitchen_table_butter_front_init_region"),
            ("On", "butter_2", "kitchen_table_butter_back_init_region"),
            (
                "On",
                "chocolate_pudding_1",
                "kitchen_table_chocolate_pudding_init_region",
            ),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
            ("Open", "wooden_cabinet_1_top_region"),
        ]
        return states

@register_mu(scene_type="kitchen")
class KitchenScene11(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
            "flat_stove": 1,
            "microwave": 1,
        }

        object_num_info = {
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.25, 0.2],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.20, 0.35],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(0, 0),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Open", "microwave_1"),
            ("Turnon", "flat_stove_1"),
            ("Open", "wooden_cabinet_1_top_region"),
        ]
        return states

@register_mu(scene_type="kitchen")
class KitchenScene12(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
            "ketchup": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.35],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(0, 0),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.03, -0.05],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.10],
                region_name="ketchup_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, -0.25],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "ketchup_1", "kitchen_table_ketchup_init_region"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Open", "microwave_1"),
        ]
        return states

@register_mu(scene_type="kitchen")
class KitchenScene13(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
            "chocolate_pudding": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.25],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.2, 0.0],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.30],
                region_name="chocolate_pudding_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.25],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Close", "microwave_1"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "chocolate_pudding_1", "kitchen_table_chocolate_pudding_init_region"),
        ]
        return states

@register_mu(scene_type="kitchen")
class KitchenScene14(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
            "flat_stove": 1,
        }

        object_num_info = {
            "white_bowl": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.20],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.10],
                region_name="white_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

@register_mu(scene_type="kitchen")
class KitchenScene15(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
            "flat_stove": 1,
        }

        object_num_info = {
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.21, 0.20],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("Turnon", "flat_stove_1"),
            ("Open", "wooden_cabinet_1_top_region"),
        ]
        return states


def main_raw():
    # kitchen_scene_1
    scene_name = "kitchen_scene1"
    language = "open the bottom drawer of the cabinet and put the bowl in it"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1"],
        goal_states=[
            ("Open", "wooden_cabinet_1_bottom_region"),
            ("In", "akita_black_bowl_1", "wooden_cabinet_1_bottom_region"),
        ],
    )

    scene_name = "kitchen_scene1"
    language = "open the top drawer of the cabinet and put the bowl in it, then close the top drawer"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1"],
        goal_states=[
            ("Close", "wooden_cabinet_1_top_region"),
            ("In", "akita_black_bowl_1", "wooden_cabinet_1_top_region"),
        ],
    )

    scene_name = "kitchen_scene1"
    language = "open the top drawer of the cabinet and put the bowl on the plate"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1", "plate_1"],
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("On", "akita_black_bowl_1", "plate_1"),
        ],
    )

    scene_name = "kitchen_scene1"
    language = "open the bottom drawer of the cabinet and put the bowl on the plate"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1", "plate_1"],
        goal_states=[
            ("Open", "wooden_cabinet_1_bottom_region"),
            ("On", "akita_black_bowl_1", "plate_1"),
        ],
    )

    # kitchen_scene_2
    scene_name = "kitchen_scene2"
    language = "open the top drawer of the cabinet, then stack the black bowl at the front on the black bowl in the middle"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1", "akita_black_bowl_2"],
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("On", "akita_black_bowl_1", "akita_black_bowl_2"),
        ],
    )

    scene_name = "kitchen_scene2"
    language = "stack the black bowl at the front on the black bowl in the middle, then stack the back black bowl on the front black bowl"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
        goal_states=[
            ("On", "akita_black_bowl_1", "akita_black_bowl_2"),
            ("On", "akita_black_bowl_3", "akita_black_bowl_1"),
        ],
    )

    scene_name = "kitchen_scene2"
    language = "stack the back black bowl on the front black bowl"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["akita_black_bowl_3"],
        goal_states=[
            ("On", "akita_black_bowl_3", "akita_black_bowl_1")
        ],
    )

    # kitchen_scene_3
    scene_name = "kitchen_scene3"
    language = "turn on the stove and put the moka pot on it"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["moka_pot_1", "flat_stove_1"],
        goal_states=[
            ("Turnon", "flat_stove_1"),
            ("On", "moka_pot_1", "flat_stove_1"),
        ],
    )

    # kitchen_scene_4
    scene_name = "kitchen_scene4"
    language = "put the black bowl on the wine rack and close the bottom drawer of the cabinet"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wine_bottle_1"],
        goal_states=[
            ("Close", "white_cabinet_1_bottom_region"),
            ("On", "wine_bottle_1", "wine_rack_1"),
        ],
    )

    scene_name = "kitchen_scene4"
    language = "put the black bowl on the wine rack and close the bottom drawer of the cabinet, then open the top drawer of the cabinet"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wine_bottle_1"],
        goal_states=[
            ("Close", "white_cabinet_1_bottom_region"),
            ("On", "wine_bottle_1", "wine_rack_1"),
            ("Open", "white_cabinet_1_top_region")
        ],
    )

    scene_name = "kitchen_scene4"
    language = "put the black bowl in the bottom drawer of the cabinet and close the bottom drawer of the cabinet"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["akita_black_bowl_1"],
        goal_states=[
            ("Close", "white_cabinet_1_bottom_region"),
            ("In", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
        ],
    ) 

    # kitchen_scene_5
    scene_name = "kitchen_scene5"
    language = "close the top drawer of the cabinet and put the black bowl on the plate"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["akita_black_bowl_1"],
        goal_states=[
            ("Close", "white_cabinet_1_top_region"),
            ("On", "akita_black_bowl_1", "plate_1"),
        ],
    ) 

    scene_name = "kitchen_scene5"
    language = "put the black bowl in the top drawer of the cabinet and close it"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["akita_black_bowl_1"],
        goal_states=[
            ("Close", "white_cabinet_1_top_region"),
            ("In", "akita_black_bowl_1", "white_cabinet_1_top_region"),
        ],
    ) 

    # kitchen_scene_6
    scene_name = "kitchen_scene6"
    language = "put the yellow and white mug in the microwave and close the microwave"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["white_yellow_mug_1", "microwave_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("In", "white_yellow_mug_1", "microwave_1"),
        ],
    ) 

    scene_name = "kitchen_scene6"
    language = "close the microwave and put the white and yellow mug to the front of the white mug"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["white_yellow_mug_1", "microwave_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("On", "white_yellow_mug_1", "kitchen_table_porcelain_mug_front_region"),
        ],
    ) 

    # kitchen_scene_7
    scene_name = "kitchen_scene7"
    language = "open the microwave and put the white bowl on the plate"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["white_bowl_1", "plate_1", "microwave_1"],
        goal_states=[
            ("Open", "microwave_1"),
            ("On", "white_bowl_1", "plate_1"),
        ],
    ) 

    scene_name = "kitchen_scene7"
    language = "open the microwave and put the white bowl in the microwave"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["white_bowl_1", "microwave_1"],
        goal_states=[
            ("In", "white_bowl_1", "microwave_1_heating_region"),
        ],
    ) 

    scene_name = "kitchen_scene7"
    language = "open the microwave and put the white bowl in the microwave, then close the microwave"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["white_bowl_1", "microwave_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("In", "white_bowl_1", "microwave_1_heating_region"),
        ],
    ) 

    # kitchen_scene_8
    scene_name = "kitchen_scene8"
    language = "put the right moka pot on the stove and turn off the stove"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["moka_pot_1", "flat_stove_1"],
        goal_states=[
            ("Turnoff", "flat_stove_1"),
            ("On", "moka_pot_1", "flat_stove_1"),
        ],
    ) 

    scene_name = "kitchen_scene8"
    language = "turn off the stove and put the right moka pot on the stove"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["moka_pot_1", "flat_stove_1"],
        goal_states=[
            ("Turnoff", "flat_stove_1"),
            ("On", "moka_pot_1", "flat_stove_1"),
        ],
    ) 

    # kitchen_scene_9
    scene_name = "kitchen_scene9"
    language = "turn on the stove and put the white bowl on the stove"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["white_bowl_1", "flat_stove_1"],
        goal_states=[
            ("Turnon", "flat_stove_1"),
            ("On", "white_bowl_1", "flat_stove_1"),
        ],
    )     

    BDDL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bddl_file_names, failures = generate_bddl_from_task_info(str(BDDL_OUTPUT_DIR))
    print(bddl_file_names)

    scene_name = "kitchen_scene11"
    language = "turn off the stove and close the top drawer of the cabinet"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["flat_stove_1"],
        goal_states=[
            ("Close", "wooden_cabinet_1_top_region"),
            ("Turnoff", "flat_stove_1")
        ],
    ) 

    scene_name = "kitchen_scene11"
    language = "turn off the stove and close the microwave"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["flat_stove_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("Turnoff", "flat_stove_1")
        ],
    ) 

    scene_name = "kitchen_scene11"
    language = "close the top drawer of the cabinet and close the microwave"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["flat_stove_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("Close", "wooden_cabinet_1_top_region")
        ],
    ) 

    # kitchen_scene_12
    scene_name = "kitchen_scene12"
    language = "close the microwave and put the black bowl on the plate"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["microwave_1", "akita_black_bowl_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("On", "akita_black_bowl_1", "plate_1"),
        ],
    )

    # kitchen_scene_12
    scene_name = "kitchen_scene12"
    language = "put the black bowl on the plate and close the microwave"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["microwave_1", "akita_black_bowl_1"],
        goal_states=[
            ("Close", "microwave_1"),
            ("On", "akita_black_bowl_1", "plate_1"),
        ],
    )

    # kitchen_scene_13
    scene_name = "kitchen_scene13"
    language = "open the microwave and put the black bowl on the plate"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["microwave_1", "akita_black_bowl_1"],
        goal_states=[
            ("Open", "microwave_1"),
            ("On", "akita_black_bowl_1", "plate_1"),
        ],
    )



def main():
    # scene_name = "kitchen_scene15"
    # language = "turn off the stove and close the drawer of the cabinet"
    # register_task_info(
    #     language,
    #     scene_name=scene_name,
    #     objects_of_interest=["flat_stove_1"],
    #     goal_states=[
    #         ("Close", "wooden_cabinet_1_top_region"),
    #         ("Turnoff", "flat_stove_1")
    #     ],
    # ) 

    scene_name = "kitchen_scene15"
    language = "close the drawer of the cabinet and turn off the stove"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["flat_stove_1"],
        goal_states=[
            ("Close", "wooden_cabinet_1_top_region"),
            ("Turnoff", "flat_stove_1")
        ],
    ) 

    # # 只生成点数最高的
    # scene_name = "kitchen_scene15"
    # language = "close the drawer and turn off the stove"
    # register_task_info(
    #     language,
    #     scene_name=scene_name,
    #     objects_of_interest=["flat_stove_1"],
    #     goal_states=[
    #         ("Close", "wooden_cabinet_1_top_region"),
    #         ("Turnoff", "flat_stove_1")
    #     ],
    # ) 

    # scene_name = "kitchen_scene15"
    # language = "turn off the stove and close the drawer"
    # register_task_info(
    #     language,
    #     scene_name=scene_name,
    #     objects_of_interest=["flat_stove_1"],
    #     goal_states=[
    #         ("Close", "wooden_cabinet_1_top_region"),
    #         ("Turnoff", "flat_stove_1")
    #     ],
    # ) 

    BDDL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bddl_file_names, failures = generate_bddl_from_task_info(str(BDDL_OUTPUT_DIR))
    print(bddl_file_names)

if __name__ == "__main__":
    main()
