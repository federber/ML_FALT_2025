import pygame
import torch
from game.wrapped_flappy_bird import GameState
from src.model import GameProcessor


class EvaluationSession:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device, weights_only=False).eval()
        self.game = GameState()

    def _prepare_initial_state(self, frame):
        """Подготавливает начальное состояние из 4 одинаковых кадров"""
        processed = GameProcessor.preprocess_frame(frame)
        tensor = GameProcessor.frame_to_tensor(processed)
        return torch.cat([tensor] * 4).unsqueeze(0).to(self.device)

    def _prepare_next_state(self, current_state, new_frame):
        """Обновляет состояние, добавляя новый кадр и удаляя самый старый"""
        processed = GameProcessor.preprocess_frame(new_frame)
        tensor = GameProcessor.frame_to_tensor(processed)
        return torch.cat([current_state.squeeze(0)[1:], tensor]).unsqueeze(0).to(self.device)

    def run(self):
        max_score = 0
        action = torch.zeros([self.model.action_space], dtype=torch.float32).to(self.device)
        action[0] = 1
        frame, _, _, score = self.game.frame_step(action)
        state = self._prepare_initial_state(frame)

        while True:
            with torch.no_grad():
                q_values = self.model(state)
                action_idx = torch.argmax(q_values)

            action = torch.zeros([self.model.action_space], dtype=torch.float32).to(self.device)
            action[action_idx] = 1

            score_past = score

            frame, _, terminal, score = self.game.frame_step(action)
            state = self._prepare_next_state(state, frame)
            max_score = max(max_score, score)

            if terminal:
                print(f"Game Over! Score: {score_past} | Max Score: {max_score}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print(f"Final Max Score: {max_score}")
                    return
