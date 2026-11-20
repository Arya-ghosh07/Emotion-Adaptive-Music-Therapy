export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5001/api';

export const MODE_INFO = {
  adaptive: {
    label: 'Adaptive',
    text: 'Starts at your heart rate and slows down; learns which style calms you.',
  },
  fixed: {
    label: 'Fixed (control)',
    text: 'One style at a constant tempo, no adaptation. For comparison.',
  },
  silence: {
    label: 'Silence (control)',
    text: 'No music. Shows how your heart rate settles on its own.',
  },
};

export const CONTEXT_LABEL = {
  elevated: 'Heart rate raised',
  near_rest: 'Near resting',
  below_rest: 'Below resting',
};
