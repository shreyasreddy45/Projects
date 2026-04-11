// ui_bloc.dart
import 'package:flutter_bloc/flutter_bloc.dart';
import 'ui_event.dart';
import 'ui_state.dart';

class UiBloc extends Bloc<UiEvent, UiState> {
  UiBloc()
      : super(UiState(
          isDrawerOpen: false,
          showFeedback: false,
          role: '',
        )) {
    on<ToggleDrawerEvent>((event, emit) {
      emit(state.copyWith(isDrawerOpen: !state.isDrawerOpen));
    });
    on<ShowFeedbackEvent>((event, emit) {
      emit(state.copyWith(showFeedback: true));
    });
    on<CloseFeedbackEvent>((event, emit) {
      emit(state.copyWith(showFeedback: false));
    });
    on<SetUserRoleEvent>((event, emit) {
      print("✅ Role set to: ${event.role}");
      emit(state.copyWith(role: event.role));
    });
  }
}
