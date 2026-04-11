import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:ProcSync/routes/app_routes.dart';
import './bloc/student_info_bloc.dart';
import './bloc/student_info_event.dart';
import './bloc/student_info_state.dart';
import 'package:ProcSync/presentation/dashboard_page/models/class_model.dart';
import 'package:ProcSync/presentation/chat/models/user_model.dart';

class StudentInfoPage extends StatelessWidget {
  final UserModel currentUser;
  final ClassModel? classModel;
  final bool isReadOnly; // NEW

  const StudentInfoPage({
    super.key,
    required this.currentUser,
    this.classModel,
    this.isReadOnly = false, // NEW
  });

  static Route route({
    required UserModel currentUser,
    ClassModel? classModel,
    bool isReadOnly = false,
  }) {
    return MaterialPageRoute(
      builder: (_) => StudentInfoPage(
        currentUser: currentUser,
        classModel: classModel,
        isReadOnly: isReadOnly,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => StudentInfoBloc(
        firestore: FirebaseFirestore.instance,
        storage: FirebaseStorage.instance,
      )..add(LoadStudentInfo(currentUser.uid)),
      child: Scaffold(
        body: SafeArea(
          child: BlocBuilder<StudentInfoBloc, StudentInfoState>(
            builder: (context, state) {
              if (state is StudentInfoLoading) {
                return const Center(child: CircularProgressIndicator());
              } else if (state is StudentInfoLoaded) {
                final info = state.studentInfo;

                return SingleChildScrollView(
                  child: Column(
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Row(
                          children: [
                            IconButton(
                              icon: const Icon(Icons.arrow_back),
                              onPressed: () => Navigator.pop(context),
                            ),
                            const Spacer(),
                            CircleAvatar(
                              radius: 20,
                              backgroundImage: NetworkImage(
                                FirebaseAuth.instance.currentUser?.photoURL ??
                                    'https://via.placeholder.com/150',
                              ),
                            ),
                          ],
                        ),
                      ),
                      Text(
                        isReadOnly
                            ? 'Student Profile Photo'
                            : 'Upload Your Profile Photo',
                        style: const TextStyle(
                            fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 20),
                      GestureDetector(
                        onTap: isReadOnly
                            ? null
                            : () async {
                                final picked = await ImagePicker()
                                    .pickImage(source: ImageSource.gallery);
                                if (picked != null) {
                                  context.read<StudentInfoBloc>().add(
                                        UploadProfilePhoto(
                                          uid: currentUser.uid,
                                          imageFile: File(picked.path),
                                        ),
                                      );
                                }
                              },
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            CircleAvatar(
                              radius: 60,
                              backgroundImage: info.profilePhotoUrl.isNotEmpty
                                  ? NetworkImage(info.profilePhotoUrl)
                                  : null,
                              backgroundColor: Colors.grey.shade300,
                              child: info.profilePhotoUrl.isEmpty
                                  ? const Icon(Icons.person,
                                      size: 60, color: Colors.white)
                                  : null,
                            ),
                            if (!isReadOnly)
                              Container(
                                width: 120,
                                height: 120,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.black.withOpacity(0.3),
                                ),
                                child: const Icon(Icons.camera_alt,
                                    color: Colors.white, size: 30),
                              ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),
                      Text(currentUser.name,
                          style: const TextStyle(fontSize: 18)),
                      Text(currentUser.email,
                          style: const TextStyle(color: Colors.grey)),
                      const SizedBox(height: 24),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            _infoButton("Personal", Icons.person, () {
                              if (!isReadOnly) {
                                Navigator.pushNamed(
                                  context,
                                  AppRoutes.personalDetailsPage,
                                  arguments: {
                                    'currentUser': currentUser,
                                    'classModel': classModel,
                                  },
                                );
                              }
                            }, isReadOnly),
                            _infoButton("Family", Icons.group, () {
                              if (!isReadOnly) {
                                Navigator.pushNamed(
                                  context,
                                  AppRoutes.familyDetailsPage,
                                  arguments: {
                                    'currentUser': currentUser,
                                    'classModel': classModel,
                                  },
                                );
                              }
                            }, isReadOnly),
                            _infoButton("Academic", Icons.school, () {
                              if (!isReadOnly) {
                                Navigator.pushNamed(
                                  context,
                                  AppRoutes.academicDetailsPage,
                                  arguments: {'userId': currentUser.uid},
                                );
                              }
                            }, isReadOnly),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              } else if (state is StudentInfoError) {
                return Center(child: Text('Error: ${state.message}'));
              }
              return const SizedBox();
            },
          ),
        ),
      ),
    );
  }

  Widget _infoButton(
      String label, IconData icon, VoidCallback onTap, bool isDisabled) {
    return Column(
      children: [
        ElevatedButton(
          style: ElevatedButton.styleFrom(
            shape: const CircleBorder(),
            padding: const EdgeInsets.all(16),
            backgroundColor: isDisabled ? Colors.grey.shade400 : null,
          ),
          onPressed: isDisabled ? null : onTap,
          child: Icon(icon),
        ),
        const SizedBox(height: 8),
        Text(label),
      ],
    );
  }
}
