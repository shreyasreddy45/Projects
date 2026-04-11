class UiState {
  final bool isDrawerOpen;
  final bool showFeedback;
  final String role;

  UiState({
    required this.isDrawerOpen,
    required this.showFeedback,
    required this.role,
  });

  UiState copyWith({bool? isDrawerOpen, bool? showFeedback, String? role}) {
    return UiState(
      isDrawerOpen: isDrawerOpen ?? this.isDrawerOpen,
      showFeedback: showFeedback ?? this.showFeedback,
      role: role ?? this.role,
    );
  }
}
