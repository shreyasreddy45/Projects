abstract class UiEvent {}

class ToggleDrawerEvent extends UiEvent {}

class ShowFeedbackEvent extends UiEvent {}

class CloseFeedbackEvent extends UiEvent {}

class SetUserRoleEvent extends UiEvent {
  final String role;
  SetUserRoleEvent(this.role);
}
