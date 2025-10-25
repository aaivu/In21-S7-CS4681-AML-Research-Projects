import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ


class MultiScaleEgoClip_EgoMCQ(EgoClip_EgoMCQ):
    """
    Multi-scale EgoClip dataset that loads videos at multiple temporal resolutions.
    
    Extends the base EgoClip_EgoMCQ dataset to provide videos sampled at different
    frame counts (4, 8, 16 frames) for multi-scale temporal modeling.
    """
    
    def __init__(self, multi_scale_frames=[4, 8, 16], temporal_stride=1, **kwargs):
        """
        Initialize multi-scale EgoClip dataset.
        
        Args:
            multi_scale_frames: List of frame counts for different scales
            temporal_stride: Stride for temporal sampling
            **kwargs: Arguments passed to parent EgoClip_EgoMCQ class
        """
        super().__init__(**kwargs)
        
        self.multi_scale_frames = multi_scale_frames
        self.temporal_stride = temporal_stride
        self.max_frames = max(multi_scale_frames)
        
        # Update video params to use max frames for video reading
        self.video_params['num_frames'] = self.max_frames
    
    def _get_video_frames_multi_scale(self, video_fp, video_sec, bound_sec):
        """
        Get video frames at multiple temporal scales.
        
        Args:
            video_fp: Video file paths
            video_sec: Video start/end seconds
            bound_sec: Boundary seconds
            
        Returns:
            multi_scale_videos: Dict containing videos at different scales
        """
        video_loading = self.video_params.get('loading', 'strict')
        
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                # Load video with maximum frames
                imgs, idxs = self.video_reader(video_fp[0], video_fp[1], self.max_frames, self.frame_sample,
                                               start_sec=video_sec[0], end_sec=video_sec[1], bound_sec=bound_sec)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        # Apply transforms
        if self.transforms is not None:
            if self.max_frames > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        # Create final tensor with max frames
        final_max = torch.zeros([self.max_frames, 3, self.video_params['input_res'],
                                self.video_params['input_res']])
        final_max[:imgs.shape[0]] = imgs
        
        # Create multi-scale versions by temporal subsampling
        multi_scale_videos = {}
        
        for num_frames in self.multi_scale_frames:
            if num_frames <= imgs.shape[0]:
                # Uniform temporal subsampling
                if num_frames == imgs.shape[0]:
                    # No subsampling needed
                    sampled_frames = final_max[:num_frames]
                else:
                    # Uniform subsampling
                    indices = np.linspace(0, imgs.shape[0] - 1, num_frames).astype(int)
                    sampled_frames = imgs[indices]
                    
                    # Pad if necessary
                    final_scale = torch.zeros([num_frames, 3, self.video_params['input_res'],
                                              self.video_params['input_res']])
                    final_scale[:sampled_frames.shape[0]] = sampled_frames
                    sampled_frames = final_scale
            else:
                # If we need more frames than available, repeat the last frame
                final_scale = torch.zeros([num_frames, 3, self.video_params['input_res'],
                                          self.video_params['input_res']])
                final_scale[:imgs.shape[0]] = imgs
                # Repeat last frame if necessary
                if imgs.shape[0] > 0:
                    final_scale[imgs.shape[0]:] = imgs[-1].unsqueeze(0)
                sampled_frames = final_scale
            
            multi_scale_videos[f'video_{num_frames}f'] = sampled_frames
        
        # Also include the original max-frame version
        multi_scale_videos['video'] = final_max
        
        return multi_scale_videos
    
    def _get_train_item(self, item):
        """
        Get training item with multi-scale videos.
        """
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, video_sec, bound_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        
        # Get multi-scale video frames
        multi_scale_videos = self._get_video_frames_multi_scale(video_fp, video_sec, bound_sec)
        
        # Scene-aware negative sampling
        if self.neg_param:
            sample_neg = self.metadata[self.metadata.segment_id==sample.segment_id].sample(1).iloc[0]
            video_fp_neg, video_sec_neg, bound_sec_neg = self._get_video_path(sample_neg)
            caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)
            multi_scale_videos_neg = self._get_video_frames_multi_scale(video_fp_neg, video_sec_neg, bound_sec_neg)

        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
        
        if self.neg_param:
            return {'video': multi_scale_videos, 'text': caption,
                    'video_neg': multi_scale_videos_neg, 'text_neg': caption_neg,
                    'meta': meta_arr,
                    'noun_vec': noun_vec, 'verb_vec': verb_vec,
                    'noun_vec_neg': noun_vec_neg, 'verb_vec_neg': verb_vec_neg}
        else:
            return {'video': multi_scale_videos, 'text': caption,
                    'meta': meta_arr,
                    'noun_vec': noun_vec, 'verb_vec': verb_vec}
    
    def _get_val_item(self, item):
        """
        Get validation item with multi-scale videos.
        """
        item = item % len(self.metadata)
        itemMCQ = self.metadata[str(item)]

        answerIndex = itemMCQ['answer']
        sampleQuery = itemMCQ['query']
        textQuery, _, _ = self._get_caption(sampleQuery)

        sampleOptions = itemMCQ['choices']
        num_options = len(sampleOptions)
        textOptions = []
        
        # Initialize multi-scale video options
        videoOptions = {}
        for num_frames in self.multi_scale_frames:
            videoOptions[f'video_{num_frames}f'] = torch.zeros([num_options, num_frames, 3, 
                                                               self.video_params['input_res'],
                                                               self.video_params['input_res']])
        videoOptions['video'] = torch.zeros([num_options, self.max_frames, 3, 
                                           self.video_params['input_res'],
                                           self.video_params['input_res']])

        for id, option in enumerate(sampleOptions):
            sampleOptioni = sampleOptions[option]
            video_fp, video_sec, bound_sec = self._get_video_path(sampleOptioni)
            caption, _, _ = self._get_caption(sampleOptioni)
            textOptions.append(caption)

            multi_scale_videos = self._get_video_frames_multi_scale(video_fp, video_sec, bound_sec)
            
            # Store each scale
            for key, video_tensor in multi_scale_videos.items():
                videoOptions[key][id] = video_tensor

        type = itemMCQ['types']  # 1 for inter; 2 for intra
        data = {'video': videoOptions, 'text': textQuery, 'text_ops': textOptions, 
                'correct': answerIndex, 'type': type}
        
        return data

    def __getitem__(self, item):
        """
        Get item based on split.
        """
        if self.split == 'train':
            return self._get_train_item(item)
        else:
            return self._get_val_item(item)


class AdjacentClipSampler:
    """
    Sampler that provides adjacent clips for temporal consistency loss.
    """
    
    def __init__(self, dataset, overlap_ratio=0.5):
        """
        Initialize adjacent clip sampler.
        
        Args:
            dataset: Multi-scale dataset
            overlap_ratio: Overlap ratio between adjacent clips (0.0 to 1.0)
        """
        self.dataset = dataset
        self.overlap_ratio = overlap_ratio
        
    def get_adjacent_clips(self, item):
        """
        Get adjacent overlapping clips for temporal consistency.
        
        Args:
            item: Dataset item index
            
        Returns:
            clip1, clip2: Two adjacent overlapping clips
        """
        # This is a simplified implementation
        # In practice, you would implement proper temporal sliding window sampling
        # based on the video metadata and desired overlap
        
        # For now, return the same clip twice (identity consistency)
        # This can be extended to actual temporal sliding
        clip = self.dataset[item]
        return clip, clip