#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
import numpy as np
from torch.nn.functional import normalize as torch_normalize
from random import randint
from utils.loss_utils import l1_loss, ssim, equilateral_regularizer, l2_loss
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.transforms import matrix_to_quaternion
from typing import Union

def remove_faces_from_single_mesh(
    mesh:Meshes, 
    faces_idx_to_keep:torch.Tensor=None,
    faces_to_keep_mask:torch.Tensor=None,
    ):
    """Returns a new Meshes object with only the faces specified by faces_idx_to_keep.
    If faces_to_keep_mask is provided instead, it is used to filter out the faces to remove.

    Args:
        mesh (Meshes): _description_
        faces_idx_to_keep (torch.Tensor): Has shape (n_faces_to_keep, ).
        faces_to_keep_mask (torch.Tensor): Has shape (n_faces, ).

    Returns:
        Meshes: _description_
    """
    assert len(mesh) == 1, "This function only works with single mesh objects."
    if (faces_to_keep_mask is None) and (faces_idx_to_keep is None):
        raise ValueError("Either faces_idx_to_keep or faces_to_keep_mask must be provided.")
    
    if faces_idx_to_keep is None:
        faces_idx_to_keep = torch.arange(0, faces_to_keep_mask.shape[0], device=faces_to_keep_mask.device)[faces_to_keep_mask]
    return mesh.submeshes([[faces_idx_to_keep]])


def remove_verts_from_single_mesh(
    mesh:Meshes, 
    verts_idx_to_keep:torch.Tensor=None,
    verts_to_keep_mask:torch.Tensor=None,
    ):
    """Returns a new Meshes object with only the vertices specified by verts_idx_to_keep.
    If verts_to_keep_mask is provided instead, it is used to filter out the vertices to remove.

    Args:
        mesh (Meshes): _description_
        verts_idx_to_keep (torch.Tensor): Has shape (n_verts_to_keep, ).
        verts_to_keep_mask (torch.Tensor): Has shape (n_verts, ).

    Returns:
        Meshes: _description_
    """
    assert len(mesh) == 1, "This function only works with single mesh objects."
    if (verts_idx_to_keep is None) and (verts_to_keep_mask is None):
        raise ValueError("Either verts_idx_to_keep or verts_to_keep_mask must be provided.")
    
    if verts_to_keep_mask is None:
        verts_to_keep_mask = torch.zeros(mesh.verts_packed().shape[0], device=verts_idx_to_keep.device, dtype=torch.bool)
        verts_to_keep_mask[verts_idx_to_keep] = True
    faces = mesh.faces_packed()
    faces_mask = verts_to_keep_mask[faces].any(dim=-1)
    return remove_faces_from_single_mesh(mesh, faces_to_keep_mask=faces_mask)

def get_manifold_meshes_from_pointmaps(
    points3d:torch.Tensor, 
    imgs:torch.Tensor, 
    masks=None,
    return_single_mesh_object=False,
    return_manifold_idx=False,
    return_verts_neighbors=False,
    return_faces_neighbors=False,
    device=None,
    ):
    """Creates a list of Meshes objects from a list of pointmaps and images.
    Masks can be provided to filter out points in the pointmaps. 

    Args:
        points3d (:torch.Tensor): Has shape (n_images, height, width, 3)
        imgs (:torch.Tensor): Has shape (n_images, height, width, 3)
        masks (:torch.Tensor): Has shape (n_images, height, width)
        return_single_mesh_object (bool, optional): If True, the meshes are joined into a single Meshes object. Defaults to False.
        device (torch.device, optional): Device for the output Meshes object. Defaults to None. 
            If using many images, we recommend passing points3d and other tensors on cpu and providing the GPU device in this variable.
            Vertices will be filtered with masks and progressively moved to the device to avoid OOM issues.

    Returns:
        Meshes: _description_
    """
    # TODO: Replace vertex colors with a UVTexture + an Image of a given input resolution.
    
    if device is None:
        device = points3d[0].device
    
    n_points_per_col = points3d[0].shape[0]
    n_points_per_row = points3d[0].shape[1]

    verts_idx = torch.arange(n_points_per_row * n_points_per_col)[..., None].to(device)
    verts_idx = verts_idx.reshape(n_points_per_col, n_points_per_row)[:-1, :-1].reshape(-1, 1)

    faces_1 = torch.cat([
        verts_idx,
        verts_idx + n_points_per_row,
        verts_idx + 1,
    ], dim=-1)
    faces_2 = torch.cat([
        verts_idx + n_points_per_row + 1,
        verts_idx + 1,
        verts_idx + n_points_per_row,
    ], dim=-1)

    faces = torch.cat([faces_1, faces_2], dim=0)

    manifolds = []
    manifold_idx = torch.zeros(0, device=device, dtype=torch.int64)
    for i_ptmap in range(len(points3d)):
        vert_features = torch.nn.functional.pad(
            imgs[i_ptmap].view(1, -1, 3).clamp(0, 1).to(device), pad=(0, 1), value=1.,
        )
        manifold = Meshes(
            verts=[points3d[i_ptmap].view(-1, 3)], 
            faces=[faces],
            textures=TexturesVertex(verts_features=vert_features)
        ).to(device)
        if (masks is not None) and masks.any().item():
            manifold = remove_verts_from_single_mesh(manifold, verts_to_keep_mask=masks[i_ptmap].to(device).view(-1))
        manifolds.append(manifold)
        manifold_idx = torch.cat([
            manifold_idx, 
            torch.full(size=(manifold.verts_packed().shape[0],), fill_value=i_ptmap, device=device, dtype=torch.int64)
        ])
    if return_single_mesh_object:
        manifolds = join_meshes_as_scene(manifolds)
        
    if return_manifold_idx:
        return manifolds, manifold_idx
    return manifolds

def get_regular_triangle_bary_coords(n:int, device='cpu'):
    """Returns regular barycentric coordinates in a triangle.
    The barycentric coordinates correspond to a regular set of N points,
    where N=n*(n+1)/2 is the n-th triangular number.

    Args:
        n (int): Level of the triangular number.
        device (str, optional): Defaults to 'cpu'.

    Returns:
        torch.Tensor: Has shape (N, 3) where N=n*(n+1)/2.
    """
    l = torch.arange(n, device=device).view(-1, 1, 1).repeat(1, n, 1) + 1
    k = torch.arange(n, device=device).view(1, -1, 1).repeat(n, 1, 1) + 1
    
    gamma = 1 - (l + 1) / (n + 2)
    alpha = k / (l + 1) * (1 - gamma)
    beta = (l - k + 1) / (l + 1) * (1 - gamma)
    mask = k[..., 0] <= l[..., 0]
    
    return torch.cat([alpha[mask], beta[mask], gamma[mask]], dim=-1)



def get_triangle_surfel_parameters_from_mesh(
    barycentric_coords:Union[torch.Tensor, int], 
    mesh:Meshes=None, 
    verts:torch.Tensor=None, faces:torch.Tensor=None, verts_features:torch.Tensor=None,
    get_colors_from_mesh:bool=False,
    get_opacity_from_mesh:bool=False,
    return_face_triangles:bool=False,
    ):
    """Get triangle parameters from mesh using barycentric coordinates.
    
    This function is adapted from get_gaussian_surfel_parameters_from_mesh but returns
    triangle vertices instead of Gaussian parameters.

    Args:
        barycentric_coords (Union[torch.Tensor, int]): Has shape (n_faces, n_triangles_per_face, 3) or (n_triangles_per_face, 3).
            If an int is provided, the regular barycentric coordinates of a triangle with the corresponding triangular number will be used.
        mesh (Meshes, optional): Defaults to None.
        verts (torch.Tensor, optional): Has shape (n_verts, 3). Defaults to None.
        faces (torch.Tensor, optional): Has shape (n_faces, 3). Defaults to None.
        verts_features (torch.Tensor, optional): Has shape (n_verts, feature_dim). Defaults to None.
        get_colors_from_mesh (bool, optional): Whether to extract colors from mesh. Defaults to False.
        get_opacity_from_mesh (bool, optional): Whether to extract opacities from mesh. Defaults to False.
        return_face_triangles (bool, optional): If True, returns the original mesh faces as triangles. Defaults to False.

    Raises:
        ValueError: Either mesh or (verts, faces) should be provided.
        
    Returns:
        dict: Contains triangle parameters including vertices, colors, and opacities.
    """
    
    mesh_is_provided = mesh is not None
    verts_and_faces_are_provided = (verts is not None) and (faces is not None)
    if not mesh_is_provided and not verts_and_faces_are_provided:
        raise ValueError("Either mesh or (verts, faces) should be provided.")
    if verts is None:
        verts = mesh.verts_packed()
    if faces is None:
        faces = mesh.faces_packed()
    if verts_features is None and mesh_is_provided:
        verts_features = mesh.textures.verts_features_packed()
    
    device = verts.device
    
    if return_face_triangles:
        # Return the original mesh faces as triangles
        faces_verts = verts[faces]  # (n_faces, 3, 3)
        triangle_vertices = faces_verts.reshape(-1, 3)  # (n_faces * 3, 3)
        
        package = {
            'triangle_vertices': triangle_vertices,  # (n_faces * 3, 3)
            'face_triangles': faces_verts,  # (n_faces, 3, 3) - original face triangles
        }
        
        if get_colors_from_mesh or get_opacity_from_mesh:
            feature_size = min(verts_features.shape[-1], 4)
            features = verts_features[:, :4][faces].reshape(-1, feature_size)  # (n_faces * 3, feature_size)

            if get_colors_from_mesh:
                package['colors'] = features[..., :3]

            if get_opacity_from_mesh:
                package['opacities'] = features[..., 3:]
        
        return package
        
    # If barycentric_coords is an int, then the regular triangle barycentric coordinates are used as default
    if isinstance(barycentric_coords, int):
        barycentric_coords = get_regular_triangle_bary_coords(barycentric_coords, device=device)
    
    # Get the triangle vertices using the barycentric coordinates and the vertices of the mesh
    n_triangles_per_face = barycentric_coords.shape[-2]
    if barycentric_coords.dim() == 2:
        bary_coords = barycentric_coords[None]  # (1, n_triangles_per_face, 3). Same barycentric coordinates will be used for all faces
    else:
        bary_coords = barycentric_coords  # (n_faces, n_triangles_per_face, 3)
    
    faces_verts = verts[faces]  # (n_faces, 3, 3)
    # For each face, generate triangle vertices using barycentric coordinates
    triangle_vertices = (bary_coords[:, :, :, None] * faces_verts[:, None]).sum(dim=-2)  # (n_faces, n_triangles_per_face, 3)
    
    # Reshape to get all triangle vertices
    triangle_vertices = triangle_vertices.reshape(-1, 3)  # (n_faces * n_triangles_per_face, 3)
    
    package = {
        'triangle_vertices': triangle_vertices,  # (n_faces * n_triangles_per_face, 3)
    }
    
    if get_colors_from_mesh or get_opacity_from_mesh:
        feature_size = min(verts_features.shape[-1], 4)
        features = (
            verts_features[:, :4][faces][:, None]  # (n_faces, 1, 3, 4)
            * bary_coords[:, :, :, None]  # (n_faces, n_triangles_per_face, 3, 1)
        ).sum(dim=-2).reshape(-1, feature_size)  # (n_faces * n_triangles_per_face, feature_size)

        if get_colors_from_mesh:
            package['colors'] = features[..., :3]

        if get_opacity_from_mesh:
            package['opacities'] = features[..., 3:]
    
    return package

def get_triangle_parameters_from_charts_data(
    charts_data: dict, 
    images, 
    conf_th=-1.,
    ratio_th=5.,
    barycentric_coords=5,
    get_colors_from_mesh=True,
    get_opacity_from_mesh=False,
    return_face_triangles=True,
):
    """Get triangle parameters from charts data.
    
    This function is adapted from get_gaussian_parameters_from_charts_data but returns
    triangle parameters instead of Gaussian parameters.

    Args:
        charts_data (dict): Contains 'pts' and 'confs' keys with point data and confidence scores.
        images: Image data for manifold mesh generation.
        conf_th (float, optional): Confidence threshold for filtering. Defaults to -1.
        ratio_th (float, optional): Ratio threshold for removing elongated faces. Defaults to 5.
        barycentric_coords (Union[torch.Tensor, int], optional): Barycentric coordinates for triangle generation. Defaults to 1.
        get_colors_from_mesh (bool, optional): Whether to extract colors from mesh. Defaults to True.
        get_opacity_from_mesh (bool, optional): Whether to extract opacities from mesh. Defaults to False.
        return_face_triangles (bool, optional): If True, returns the original mesh faces as triangles. Defaults to True.

    Returns:
        dict: Triangle parameters including vertices, colors, and opacities.
    """
    charts_pts = charts_data['pts'] / charts_data['scale_factor']
    charts_confs = charts_data['confs']
    
    print("Conf Max/min: ", charts_confs.max(), charts_confs.min())
    
    # Get manifold mesh and remove faces with low confidence if needed
    manifold = get_manifold_meshes_from_pointmaps(
        points3d=charts_pts,
        imgs=images, 
        masks=charts_confs > conf_th,  
        return_single_mesh_object=True
    )
    
    # Remove elongated faces
    faces_verts = manifold.verts_packed()[manifold.faces_packed()]  # (n_faces, 3, 3)
    sides = (
        torch.roll(faces_verts, 1, dims=1)  # C, A, B
        - faces_verts  # A, B, C
    )  # (n_faces, 3, 3)  ;  AC, BA, CB
    normalized_sides = torch.nn.functional.normalize(sides, dim=-1)  # (n_faces, 3, 3)  ;  AC/||AC||, BA/||BA||, CB/||CB||
    alts = (
        sides  # AC
        - (sides * torch.roll(normalized_sides, -1, dims=1)).sum(dim=-1, keepdim=True) * normalized_sides # - (AC . BA) BA / ||BA||^2
    )  # (n_faces, 3, 3)
    alt_lengths = alts.norm(dim=-1)
    alt_ratios = alt_lengths.max(dim=1).values / alt_lengths.min(dim=1).values
    faces_mask = alt_ratios < ratio_th
    manifold = remove_faces_from_single_mesh(manifold, faces_to_keep_mask=faces_mask)
    
    # Get triangle parameters
    triangle_params = get_triangle_surfel_parameters_from_mesh(
        barycentric_coords=barycentric_coords,
        mesh=manifold,
        get_colors_from_mesh=get_colors_from_mesh,
        get_opacity_from_mesh=get_opacity_from_mesh,
        return_face_triangles=return_face_triangles,
    )
    
    return triangle_params

def load_charts_data(path: str, device:str='cuda'):
    """Load charts data from a .npz file created by save_charts_data.
    
    Args:
        path (str): Path to the .npz file containing the charts data
        
    Returns:
        dict: Dictionary containing the loaded data with keys:
            - pts: 3D points (if saved)
            - cols: Point colors (if saved)
            - confs: Confidence values (if saved)
            - depths: Depth values (if saved)
            - normals: Normal vectors (if saved) 
            - curvatures: Curvature values (if saved)
    """
    if not path.endswith('.npz'):
        path = path + '.npz'
        
    data = np.load(path)
    
    # Convert all arrays to torch tensors
    output = {}
    for key in data.files:
        if data[key] is not None:
            output[key] = torch.from_numpy(data[key]).to(device)
            
    return output

def training(
        dataset,   
        opt, 
        pipe,
        no_dome, 
        outdoor,
        testing_iterations,
        save_iterations,
        checkpoint, 
        debug_from,
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    opt.print()
    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points, opt.set_sigma, no_dome)
    triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma, opt.lr_triangles_points_init)
    create_triangles_from_charts_data = True
    if create_triangles_from_charts_data:
        charts_data_path = f'{dataset.source_path}/charts_data.npz'
        charts_data = load_charts_data(charts_data_path)
        h_charts, w_charts = charts_data['pts'].shape[-3:-1]
        _images = [
            torch.nn.functional.interpolate(cam.original_image[None].cuda(), (h_charts, w_charts), mode="bilinear", antialias=True)[0].permute(1, 2, 0)
            for cam in scene.getTrainCameras()
        ]
        triangle_params = get_triangle_parameters_from_charts_data(
            charts_data=charts_data, 
            images=_images, 
            conf_th=-1.,  # TODO: Try higher values
            ratio_th=5.,
            barycentric_coords=1,
            get_colors_from_mesh=True,
            get_opacity_from_mesh=False,
            return_face_triangles=True,
        )

        triangles.create_from_charts(
            triangle_params=triangle_params,
            spatial_lr_scale=scene.cameras_extent,
            opacity=opt.set_opacity,
            init_size=opt.triangle_size,
            nb_points=opt.nb_points,
            set_sigma=opt.set_sigma,
            no_dome=no_dome
        )
        scene = Scene(dataset, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points, opt.set_sigma, no_dome)
        triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma, opt.lr_triangles_points_init)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    total_dead = 0

    opacity_now = True

    new_round = False
    removed_them = False

    large_scene = triangles.large

    if large_scene and outdoor:
        loss_fn = l2_loss
    else:
        loss_fn = l1_loss

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        triangles.update_learning_rate(iteration)
        if iteration % 500 == 0:
            print(len(triangles._triangles_points))

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # largest distance from point to center of image
        triangle_area = render_pkg["density_factor"].detach()
        # largest distance from point after applying sigma to center of image
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        pixel_loss = loss_fn(image, gt_image)

        ##############################################################
        # WE ADD A LOSS FORCING LOW OPACITIES                        #
        ##############################################################
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # loss opacity
        loss_opacity = torch.abs(triangles.get_opacity).mean() * args.lambda_opacity

        # loss normal and distortion
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0 # 0.001
        rend_dist = render_pkg["rend_dist"]
        dist_loss = lambda_dist * (rend_dist).mean()
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean() 
        loss_size = loss_size * opt.lambda_size


        if iteration < opt.densify_until_iter:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_size
        else:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + torch.abs(triangles.get_sigma).mean() + 1 / (torch.abs(triangles.get_opacity).mean() * 0.05) # we push the triangles towards being more opaque and solid

        loss.backward()
     
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in save_iterations:
                print("\n[ITER {}] Saving Triangles".format(iteration))
                scene.save(iteration)
            if iteration % 1000 == 0:
                total_dead = 0

            if iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0 and iteration > opt.densify_from_iter:
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()

                if iteration > 1000 and not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                    
                    if not outdoor:
                        mask_test = triangles.image_size > 1400
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                          

                total_dead += dead_mask.sum()

                if opt.proba_distr == 0:
                    oddGroup = True
                elif opt.proba_distr == 1:
                    oddGroup = False
                else:
                    if opacity_now:
                        oddGroup = opacity_now
                        opacity_now = False
                    else:
                        oddGroup = opacity_now
                        opacity_now = True

                removed_them = True
                new_round = False

                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=oddGroup, dead_mask=dead_mask)


            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                # We increase the opacity threshold to remove more triangles that are transparent
                args.opacity_dead = 0.1
                if iteration > 6000:
                    args.opacity_dead = 0.7

                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    # if we did not iterate over all the views, we do not want to remove the triangles
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()

                if iteration > 6000:
                    mask_test = triangles.get_sigma >= 1.0 # we remove the triangles that are not solid enough
                    dead_mask =  torch.logical_or(dead_mask, mask_test.squeeze())

                if not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                triangles.remove_final_points(dead_mask)
                removed_them = True
                new_round = False

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                
    print("Training is done")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.triangles.get_triangles_points.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.no_dome,
             args.outdoor,
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             )
    
    # All done
    print("\nTraining complete.")