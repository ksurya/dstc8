# DSTC8

dialogue_id
services
turns_frames_actions_act
turns_frames_actions_slot
turns_frames_actions_values
turns_frames_service
turns_frames_slots_exclusive_end
turns_frames_slots_slot
turns_frames_slots_start
turns_frames_state_active_intent
turns_frames_state_requested_slots
turns_frames_state_slot_values
turns_speaker
turns_utterance

field                      dimension

dialogue_id                [batch, 1]
services                   [batch, services]
speaker                    [batch, turns, speaker-onehot]
utterance                  [batch, turns, utterance-tokens]
actions_act                [batch, turns, frames, acts-count, acts-onehot]
actions_slot               [batch, turns, frames, acts-count, slot-onehot]
actions_values             [batch, turns, frames, acts-count, vals-count, val-tokens]
frames_service             [batch, turns, frames, service-onehot]
slots_slot                 [batch, turns, frames, slots-count, slot-onehot]
slots_start                [batch, turns, frames, slots-count]
slots_end                  [batch, turns, frames, slots-count]
active_intent              [batch, turns, frames, intent-count, intent-onehot]
requested_slots            [batch, turns, frames, slots-count, slot-onehot]
slot_values                [batch, turns, frames, slots-count, vals-count, val-tokens]
